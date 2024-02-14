//! Ledger's state storage with key-value backed store and a merkle tree

pub mod write_log;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::format;
use std::iter::Peekable;
use std::ops::{Deref, DerefMut};

use namada_core::address::{Address, EstablishedAddressGen, InternalAddress};
use namada_core::borsh::{BorshDeserialize, BorshSerialize, BorshSerializeExt};
use namada_core::chain::{ChainId, CHAIN_ID_LENGTH};
use namada_core::eth_bridge_pool::is_pending_transfer_key;
use namada_core::hash::{Error as HashError, Hash};
pub use namada_core::hash::{Sha256Hasher, StorageHasher};
pub use namada_core::storage::{
    BlockHash, BlockHeight, BlockResults, Epoch, Epochs, EthEventsQueue,
    Header, Key, KeySeg, TxIndex, BLOCK_HASH_LENGTH, BLOCK_HEIGHT_LENGTH,
    EPOCH_TYPE_LENGTH,
};
use namada_core::tendermint::merkle::proof::ProofOps;
use namada_core::time::DateTimeUtc;
use namada_core::validity_predicate::VpSentinel;
use namada_core::{encode, ethereum_structs, storage};
use namada_gas::{
    GasMetering, TxGasMeter, VpGasMeter, MEMORY_ACCESS_GAS_PER_BYTE,
    STORAGE_ACCESS_GAS_PER_BYTE, STORAGE_WRITE_GAS_PER_BYTE,
};
pub use namada_merkle_tree::{
    self as merkle_tree, ics23_specs, MembershipProof, MerkleTree,
    MerkleTreeStoresRead, MerkleTreeStoresWrite, StoreRef, StoreType,
};
use namada_merkle_tree::{Error as MerkleTreeError, MerkleRoot};
use namada_parameters::{self, EpochDuration, Parameters};
use namada_replay_protection as replay_protection;
pub use namada_storage::conversion_state::{
    ConversionState, WithConversionState,
};
pub use namada_storage::{Error as StorageError, Result as StorageResult, *};
use namada_tx::data::TxSentinel;
use thiserror::Error;
use tx_queue::{ExpiredTxsQueue, TxQueue};
use write_log::{ReProtStorageModification, StorageModification, WriteLog};

/// A result of a function that may fail
pub type Result<T> = std::result::Result<T, Error>;

/// We delay epoch change 2 blocks to keep it in sync with Tendermint, because
/// it has 2 blocks delay on validator set update.
pub const EPOCH_SWITCH_BLOCKS_DELAY: u32 = 2;

/// Owned state with full R/W access.
#[derive(Debug)]
pub struct FullAccessState<D, H>(WlState<D, H>)
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher;

/// Common trait for read-only access to write log, DB and in-memory state.
pub trait StateRead: StorageRead + Debug {
    /// DB type
    type D: 'static + DB + for<'iter> DBIter<'iter>;
    /// DB hasher type
    type H: 'static + StorageHasher;

    /// Borrow `WriteLog`
    fn write_log(&self) -> &WriteLog;

    /// Borrow `DB`
    fn db(&self) -> &Self::D;

    /// Borrow `InMemory` state
    fn in_mem(&self) -> &InMemory<Self::H>;

    fn charge_gas(&self, gas: u64) -> Result<()>;

    // TODO: the storage methods are taken from `State`, but they are not in the
    // right place - they ignore write log and only access the DB.
    /// Check if the given key is present in storage. Returns the result and the
    /// gas cost.
    fn db_has_key(&self, key: &storage::Key) -> Result<(bool, u64)> {
        Ok((
            self.db().read_subspace_val(key)?.is_some(),
            key.len() as u64 * STORAGE_ACCESS_GAS_PER_BYTE,
        ))
    }

    /// Returns a value from the specified subspace and the gas cost
    fn db_read(&self, key: &storage::Key) -> Result<(Option<Vec<u8>>, u64)> {
        tracing::debug!("storage read key {}", key);

        match self.db().read_subspace_val(key)? {
            Some(v) => {
                let gas =
                    (key.len() + v.len()) as u64 * STORAGE_ACCESS_GAS_PER_BYTE;
                Ok((Some(v), gas))
            }
            None => Ok((None, key.len() as u64 * STORAGE_ACCESS_GAS_PER_BYTE)),
        }
    }

    /// WARNING: This only works for values that have been committed to DB.
    /// To be able to see values written or deleted, but not yet committed,
    /// use the `StorageWithWriteLog`.
    ///
    /// Returns a prefix iterator, ordered by storage keys, and the gas cost.
    fn db_iter_prefix(
        &self,
        prefix: &Key,
    ) -> (<Self::D as DBIter<'_>>::PrefixIter, u64) {
        (
            self.db().iter_prefix(Some(prefix)),
            prefix.len() as u64 * STORAGE_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Returns an iterator over the block results
    fn db_iter_results(&self) -> (<Self::D as DBIter<'_>>::PrefixIter, u64) {
        (self.db().iter_results(), 0)
    }

    /// Get the hash of a validity predicate for the given account address and
    /// the gas cost for reading it.
    fn validity_predicate(
        &self,
        addr: &Address,
    ) -> Result<(Option<Hash>, u64)> {
        let key = if let Address::Implicit(_) = addr {
            namada_parameters::storage::get_implicit_vp_key()
        } else {
            Key::validity_predicate(addr)
        };
        match self.db_read(&key)? {
            (Some(value), gas) => {
                let vp_code_hash = Hash::try_from(&value[..])
                    .map_err(Error::InvalidCodeHash)?;
                Ok((Some(vp_code_hash), gas))
            }
            (None, gas) => Ok((None, gas)),
        }
    }

    /// Get the block header
    fn get_block_header(
        &self,
        height: Option<BlockHeight>,
    ) -> Result<(Option<Header>, u64)> {
        match height {
            Some(h) if h == self.in_mem().get_block_height().0 => {
                let header = self.in_mem().header.clone();
                let gas = match header {
                    Some(ref header) => {
                        header.encoded_len() as u64 * MEMORY_ACCESS_GAS_PER_BYTE
                    }
                    None => MEMORY_ACCESS_GAS_PER_BYTE,
                };
                Ok((header, gas))
            }
            Some(h) => match self.db().read_block_header(h)? {
                Some(header) => {
                    let gas = header.encoded_len() as u64
                        * STORAGE_ACCESS_GAS_PER_BYTE;
                    Ok((Some(header), gas))
                }
                None => Ok((None, STORAGE_ACCESS_GAS_PER_BYTE)),
            },
            None => {
                Ok((self.in_mem().header.clone(), STORAGE_ACCESS_GAS_PER_BYTE))
            }
        }
    }
}

/// Common trait for write log, DB and in-memory state.
pub trait State: StateRead + StorageWrite {
    /// Borrow mutable `WriteLog`
    fn write_log_mut(&mut self) -> &mut WriteLog;

    /// Splitting borrow to get mutable reference to `WriteLog`, immutable
    /// reference to the `InMemory` state and DB when in need of both (avoids
    /// complain from the borrow checker)
    fn split_borrow(&mut self)
    -> (&mut WriteLog, &InMemory<Self::H>, &Self::D);

    /// Write the provided tx hash to write log.
    fn write_tx_hash(&mut self, hash: Hash) -> write_log::Result<()> {
        self.write_log_mut().write_tx_hash(hash)
    }
}

impl<D, H> StateRead for FullAccessState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    type D = D;
    type H = H;

    fn db(&self) -> &D {
        &self.0.db
    }

    fn in_mem(&self) -> &InMemory<Self::H> {
        &self.0.in_mem
    }

    fn write_log(&self) -> &WriteLog {
        &self.0.write_log
    }

    fn charge_gas(&self, _gas: u64) -> Result<()> {
        Ok(())
    }
}

impl<D, H> State for FullAccessState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.0.write_log
    }

    fn split_borrow(
        &mut self,
    ) -> (&mut WriteLog, &InMemory<Self::H>, &Self::D) {
        (&mut self.0.write_log, &self.0.in_mem, &self.0.db)
    }
}

impl<D, H> WithConversionState for FullAccessState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    fn conversion_state(&self) -> &ConversionState {
        &self.in_mem().conversion_state
    }

    fn conversion_state_mut(&mut self) -> &mut ConversionState {
        &mut self.in_mem_mut().conversion_state
    }
}

impl<D, H> StateRead for WlState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    type D = D;
    type H = H;

    fn write_log(&self) -> &WriteLog {
        &self.write_log
    }

    fn db(&self) -> &D {
        &self.db
    }

    fn in_mem(&self) -> &InMemory<Self::H> {
        &self.in_mem
    }

    fn charge_gas(&self, _gas: u64) -> Result<()> {
        Ok(())
    }
}

impl<D, H> State for WlState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.write_log
    }

    fn split_borrow(
        &mut self,
    ) -> (&mut WriteLog, &InMemory<Self::H>, &Self::D) {
        (&mut self.write_log, &self.in_mem, &self.db)
    }
}

impl<D, H> StateRead for TempWlState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    type D = D;
    type H = H;

    fn write_log(&self) -> &WriteLog {
        &self.write_log
    }

    fn db(&self) -> &D {
        self.db
    }

    fn in_mem(&self) -> &InMemory<Self::H> {
        self.in_mem
    }

    fn charge_gas(&self, _gas: u64) -> Result<()> {
        Ok(())
    }
}

impl<D, H> State for TempWlState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.write_log
    }

    fn split_borrow(
        &mut self,
    ) -> (&mut WriteLog, &InMemory<Self::H>, &Self::D) {
        (&mut self.write_log, (self.in_mem), (self.db))
    }
}

impl<D, H> StateRead for TxHostEnvState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    type D = D;
    type H = H;

    fn write_log(&self) -> &WriteLog {
        self.write_log
    }

    fn db(&self) -> &D {
        self.db
    }

    fn in_mem(&self) -> &InMemory<Self::H> {
        self.in_mem
    }

    fn charge_gas(&self, gas: u64) -> Result<()> {
        self.gas_meter.borrow_mut().consume(gas).map_err(|err| {
            self.sentinel.borrow_mut().set_out_of_gas();
            tracing::info!(
                "Stopping transaction execution because of gas error: {}",
                err
            );
            Error::Gas(err)
        })
    }
}

impl<D, H> State for TxHostEnvState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    fn write_log_mut(&mut self) -> &mut WriteLog {
        self.write_log
    }

    fn split_borrow(
        &mut self,
    ) -> (&mut WriteLog, &InMemory<Self::H>, &Self::D) {
        (self.write_log, (self.in_mem), (self.db))
    }
}

impl<D, H> StateRead for VpHostEnvState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    type D = D;
    type H = H;

    fn write_log(&self) -> &WriteLog {
        self.write_log
    }

    fn db(&self) -> &D {
        self.db
    }

    fn in_mem(&self) -> &InMemory<Self::H> {
        self.in_mem
    }

    fn charge_gas(&self, gas: u64) -> Result<()> {
        self.gas_meter.borrow_mut().consume(gas).map_err(|err| {
            self.sentinel.borrow_mut().set_out_of_gas();
            tracing::info!(
                "Stopping VP execution because of gas error: {}",
                err
            );
            Error::Gas(err)
        })
    }
}

/// State with a write-logged storage.
#[derive(Debug)]
pub struct WlState<D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    /// Write log
    write_log: WriteLog,
    // DB (usually a MockDB or PersistentDB)
    // This should be immutable in WlState, but mutable in `FullAccess`.
    // TODO: maybe now can use &D and shit can be public? Since host_env is
    // using `trait State`.
    db: D,
    /// State in memory
    in_mem: InMemory<H>,
    /// Static merkle tree storage key filter
    pub merkle_tree_key_filter: fn(&storage::Key) -> bool,
}

/// State with a temporary write log. This is used for dry-running txs and ABCI
/// prepare and processs proposal, which must not modify the actual state.
#[derive(Debug)]
pub struct TempWlState<'a, D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    /// Write log
    write_log: WriteLog,
    // DB
    db: &'a D,
    /// State
    in_mem: &'a InMemory<H>,
}

// State with mutable write log and gas metering for tx host env.
#[derive(Debug)]
pub struct TxHostEnvState<'a, D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    /// Write log
    pub write_log: &'a mut WriteLog,
    // DB
    pub db: &'a D,
    /// State
    pub in_mem: &'a InMemory<H>,
    /// Tx gas meter
    pub gas_meter: &'a RefCell<TxGasMeter>,
    /// Errors sentinel
    pub sentinel: &'a RefCell<TxSentinel>,
}

// Read-only state with gas metering for VP host env.
#[derive(Debug)]
pub struct VpHostEnvState<'a, D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    /// Write log
    pub write_log: &'a WriteLog,
    // DB
    pub db: &'a D,
    /// State
    pub in_mem: &'a InMemory<H>,
    /// VP gas meter
    pub gas_meter: &'a RefCell<VpGasMeter>,
    /// Errors sentinel
    pub sentinel: &'a RefCell<VpSentinel>,
}

impl<D, H> FullAccessState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    pub fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.0.write_log
    }

    pub fn in_mem_mut(&mut self) -> &mut InMemory<H> {
        &mut self.0.in_mem
    }

    pub fn db_mut(&mut self) -> &mut D {
        &mut self.0.db
    }

    pub fn restrict_writes_to_write_log(&mut self) -> &mut WlState<D, H> {
        &mut self.0
    }

    pub fn read_only(&self) -> &WlState<D, H> {
        &self.0
    }

    pub fn open(
        db_path: impl AsRef<std::path::Path>,
        cache: Option<&D::Cache>,
        chain_id: ChainId,
        native_token: Address,
        storage_read_past_height_limit: Option<u64>,
        merkle_tree_key_filter: fn(&storage::Key) -> bool,
    ) -> Self {
        let write_log = WriteLog::default();
        let db = D::open(db_path, cache);
        let in_mem = InMemory::new(
            chain_id,
            native_token,
            storage_read_past_height_limit,
        );
        let mut state = Self(WlState {
            write_log,
            db,
            in_mem,
            merkle_tree_key_filter,
        });
        state.load_last_state();
        state
    }

    #[allow(dead_code)]
    /// Check if the given address exists on chain and return the gas cost.
    pub fn db_exists(&self, addr: &Address) -> Result<(bool, u64)> {
        let key = Key::validity_predicate(addr);
        self.db_has_key(&key)
    }

    /// Initialize a new epoch when the current epoch is finished. Returns
    /// `true` on a new epoch.
    pub fn update_epoch(
        &mut self,
        height: BlockHeight,
        time: DateTimeUtc,
    ) -> StorageResult<bool> {
        let parameters = namada_parameters::read(self)
            .expect("Couldn't read protocol parameters");

        match self.in_mem.update_epoch_blocks_delay.as_mut() {
            None => {
                // Check if the new epoch minimum start height and start time
                // have been fulfilled. If so, queue the next
                // epoch to start two blocks into the future so
                // as to align validator set updates + etc with
                // tendermint. This is because tendermint has a two block delay
                // to validator changes.
                let current_epoch_duration_satisfied = height
                    >= self.in_mem.next_epoch_min_start_height
                    && time >= self.in_mem.next_epoch_min_start_time;
                if current_epoch_duration_satisfied {
                    self.in_mem.update_epoch_blocks_delay =
                        Some(EPOCH_SWITCH_BLOCKS_DELAY);
                }
            }
            Some(blocks_until_switch) => {
                *blocks_until_switch -= 1;
            }
        };
        let new_epoch =
            matches!(self.in_mem.update_epoch_blocks_delay, Some(0));

        if new_epoch {
            // Reset the delay tracker
            self.in_mem.update_epoch_blocks_delay = None;

            // Begin a new epoch
            self.in_mem.block.epoch = self.in_mem.block.epoch.next();
            let EpochDuration {
                min_num_of_blocks,
                min_duration,
            } = parameters.epoch_duration;
            self.in_mem.next_epoch_min_start_height =
                height + min_num_of_blocks;
            self.in_mem.next_epoch_min_start_time = time + min_duration;

            self.in_mem.block.pred_epochs.new_epoch(height);
            tracing::info!("Began a new epoch {}", self.in_mem.block.epoch);
        }
        Ok(new_epoch)
    }

    /// Commit the current block's write log to the storage and commit the block
    /// to DB. Starts a new block write log.
    pub fn commit_block(&mut self) -> StorageResult<()> {
        if self.in_mem.last_epoch != self.in_mem.block.epoch {
            self.in_mem_mut()
                .update_epoch_in_merkle_tree()
                .into_storage_result()?;
        }

        let mut batch = D::batch();
        self.commit_write_log_block(&mut batch)
            .into_storage_result()?;
        self.commit_block_from_batch(batch).into_storage_result()
    }

    /// Commit the current block's write log to the storage. Starts a new block
    /// write log.
    pub fn commit_write_log_block(
        &mut self,
        batch: &mut D::WriteBatch,
    ) -> Result<()> {
        for (key, entry) in
            std::mem::take(&mut self.0.write_log.block_write_log).into_iter()
        {
            match entry {
                StorageModification::Write { value } => {
                    self.batch_write_subspace_val(batch, &key, value)?;
                }
                StorageModification::Delete => {
                    self.batch_delete_subspace_val(batch, &key)?;
                }
                StorageModification::InitAccount { vp_code_hash } => {
                    self.batch_write_subspace_val(batch, &key, vp_code_hash)?;
                }
                // temporary value isn't persisted
                StorageModification::Temp { .. } => {}
            }
        }
        debug_assert!(self.0.write_log.block_write_log.is_empty());

        // Replay protections specifically
        for (hash, entry) in
            std::mem::take(&mut self.0.write_log.replay_protection).into_iter()
        {
            match entry {
                ReProtStorageModification::Write => self
                    .write_replay_protection_entry(
                        batch,
                        // Can only write tx hashes to the previous block, no
                        // further
                        &replay_protection::last_key(&hash),
                    )?,
                ReProtStorageModification::Delete => self
                    .delete_replay_protection_entry(
                        batch,
                        // Can only delete tx hashes from the previous block,
                        // no further
                        &replay_protection::last_key(&hash),
                    )?,
                ReProtStorageModification::Finalize => {
                    self.write_replay_protection_entry(
                        batch,
                        &replay_protection::all_key(&hash),
                    )?;
                    self.delete_replay_protection_entry(
                        batch,
                        &replay_protection::last_key(&hash),
                    )?;
                }
            }
        }
        debug_assert!(self.0.write_log.replay_protection.is_empty());

        if let Some(address_gen) = self.0.write_log.address_gen.take() {
            self.0.in_mem.address_gen = address_gen
        }
        Ok(())
    }

    /// Start write batch.
    pub fn batch() -> D::WriteBatch {
        D::batch()
    }

    /// Execute write batch.
    pub fn exec_batch(&mut self, batch: D::WriteBatch) -> Result<()> {
        Ok(self.db.exec_batch(batch)?)
    }

    /// Batch write the value with the given height and account subspace key to
    /// the DB. Returns the size difference from previous value, if any, or
    /// the size of the value otherwise.
    pub fn batch_write_subspace_val(
        &mut self,
        batch: &mut D::WriteBatch,
        key: &Key,
        value: impl AsRef<[u8]>,
    ) -> Result<i64> {
        let value = value.as_ref();
        let is_key_merklized = (self.merkle_tree_key_filter)(key);

        if is_pending_transfer_key(key) {
            // The tree of the bridge pool stores the current height for the
            // pending transfer
            let height = self.in_mem.block.height.serialize_to_vec();
            self.in_mem.block.tree.update(key, height)?;
        } else {
            // Update the merkle tree
            if is_key_merklized {
                self.in_mem.block.tree.update(key, value)?;
            }
        }
        Ok(self.db.batch_write_subspace_val(
            batch,
            self.in_mem.block.height,
            key,
            value,
            is_key_merklized,
        )?)
    }

    /// Batch delete the value with the given height and account subspace key
    /// from the DB. Returns the size of the removed value, if any, 0 if no
    /// previous value was found.
    pub fn batch_delete_subspace_val(
        &mut self,
        batch: &mut D::WriteBatch,
        key: &Key,
    ) -> Result<i64> {
        let is_key_merklized = (self.merkle_tree_key_filter)(key);
        // Update the merkle tree
        if is_key_merklized {
            self.in_mem.block.tree.delete(key)?;
        }
        Ok(self.db.batch_delete_subspace_val(
            batch,
            self.in_mem.block.height,
            key,
            is_key_merklized,
        )?)
    }

    // Prune merkle tree stores. Use after updating self.block.height in the
    // commit.
    fn prune_merkle_tree_stores(
        &mut self,
        batch: &mut D::WriteBatch,
    ) -> Result<()> {
        if self.in_mem.block.epoch.0 == 0 {
            return Ok(());
        }
        // Prune non-provable stores at the previous epoch
        for st in StoreType::iter_non_provable() {
            self.0.db.prune_merkle_tree_store(
                batch,
                st,
                self.in_mem.block.epoch.prev(),
            )?;
        }
        // Prune provable stores
        let oldest_epoch = self.in_mem.get_oldest_epoch();
        if oldest_epoch.0 > 0 {
            // Remove stores at the previous epoch because the Merkle tree
            // stores at the starting height of the epoch would be used to
            // restore stores at a height (> oldest_height) in the epoch
            for st in StoreType::iter_provable() {
                self.db.prune_merkle_tree_store(
                    batch,
                    st,
                    oldest_epoch.prev(),
                )?;
            }

            // Prune the BridgePool subtree stores with invalid nonce
            let mut epoch = match self.get_oldest_epoch_with_valid_nonce()? {
                Some(epoch) => epoch,
                None => return Ok(()),
            };
            while oldest_epoch < epoch {
                epoch = epoch.prev();
                self.db.prune_merkle_tree_store(
                    batch,
                    &StoreType::BridgePool,
                    epoch,
                )?;
            }
        }

        Ok(())
    }

    /// Check it the given transaction's hash is already present in storage
    pub fn has_replay_protection_entry(&self, hash: &Hash) -> Result<bool> {
        Ok(self.db.has_replay_protection_entry(hash)?)
    }

    /// Write the provided tx hash to storage
    pub fn write_replay_protection_entry(
        &mut self,
        batch: &mut D::WriteBatch,
        key: &Key,
    ) -> Result<()> {
        self.db.write_replay_protection_entry(batch, key)?;
        Ok(())
    }

    /// Delete the provided tx hash from storage
    pub fn delete_replay_protection_entry(
        &mut self,
        batch: &mut D::WriteBatch,
        key: &Key,
    ) -> Result<()> {
        self.db.delete_replay_protection_entry(batch, key)?;
        Ok(())
    }

    /// Iterate the replay protection storage from the last block
    pub fn iter_replay_protection(
        &self,
    ) -> Box<dyn Iterator<Item = Hash> + '_> {
        Box::new(self.db.iter_replay_protection().map(|(raw_key, _, _)| {
            raw_key.parse().expect("Failed hash conversion")
        }))
    }

    /// Get oldest epoch which has the valid signed nonce of the bridge pool
    fn get_oldest_epoch_with_valid_nonce(&self) -> Result<Option<Epoch>> {
        let last_height = self.in_mem.get_last_block_height();
        let current_nonce = match self
            .db
            .read_bridge_pool_signed_nonce(last_height, last_height)?
        {
            Some(nonce) => nonce,
            None => return Ok(None),
        };
        let (mut epoch, _) = self.in_mem.get_last_epoch();
        // We don't need to check the older epochs because their Merkle tree
        // snapshots have been already removed
        let oldest_epoch = self.in_mem.get_oldest_epoch();
        // Look up the last valid epoch which has the previous nonce of the
        // current one. It has the previous nonce, but it was
        // incremented during the epoch.
        while 0 < epoch.0 && oldest_epoch <= epoch {
            epoch = epoch.prev();
            let height = match self
                .in_mem
                .block
                .pred_epochs
                .get_start_height_of_epoch(epoch)
            {
                Some(h) => h,
                None => continue,
            };
            let nonce = match self
                .db
                .read_bridge_pool_signed_nonce(height, last_height)?
            {
                Some(nonce) => nonce,
                // skip pruning when the old epoch doesn't have the signed nonce
                None => break,
            };
            if nonce < current_nonce {
                break;
            }
        }
        Ok(Some(epoch))
    }

    /// Rebuild full Merkle tree after [`read_last_block()`]
    fn rebuild_full_merkle_tree(
        &self,
        height: BlockHeight,
    ) -> Result<MerkleTree<H>> {
        self.get_merkle_tree(height, None)
    }

    /// Load the full state at the last committed height, if any. Returns the
    /// Merkle root hash and the height of the committed block.
    fn load_last_state(&mut self) {
        if let Some(BlockStateRead {
            merkle_tree_stores,
            hash,
            height,
            time,
            epoch,
            pred_epochs,
            next_epoch_min_start_height,
            next_epoch_min_start_time,
            update_epoch_blocks_delay,
            results,
            address_gen,
            conversion_state,
            tx_queue,
            ethereum_height,
            eth_events_queue,
        }) = self
            .0
            .db
            .read_last_block()
            .expect("Read block call must not fail")
        {
            // Rebuild Merkle tree
            let tree = MerkleTree::new(merkle_tree_stores)
                .or_else(|_| self.rebuild_full_merkle_tree(height))
                .unwrap();

            let in_mem = &mut self.0.in_mem;
            in_mem.block.hash = hash.clone();
            in_mem.block.height = height;
            in_mem.block.epoch = epoch;
            in_mem.block.results = results;
            in_mem.block.pred_epochs = pred_epochs;
            in_mem.last_block = Some(LastBlock { height, hash, time });
            in_mem.last_epoch = epoch;
            in_mem.next_epoch_min_start_height = next_epoch_min_start_height;
            in_mem.next_epoch_min_start_time = next_epoch_min_start_time;
            in_mem.update_epoch_blocks_delay = update_epoch_blocks_delay;
            in_mem.address_gen = address_gen;
            in_mem.block.tree = tree;
            in_mem.conversion_state = conversion_state;
            in_mem.tx_queue = tx_queue;
            in_mem.ethereum_height = ethereum_height;
            in_mem.eth_events_queue = eth_events_queue;
            tracing::debug!("Loaded storage from DB");
        } else {
            tracing::info!("No state could be found");
        }
    }

    /// Persist the block's state from batch writes to the database.
    /// Note that unlike `commit_block` this method doesn't commit the write
    /// log.
    pub fn commit_block_from_batch(
        &mut self,
        mut batch: D::WriteBatch,
    ) -> Result<()> {
        // All states are written only when the first height or a new epoch
        let is_full_commit = self.in_mem.block.height.0 == 1
            || self.in_mem.last_epoch != self.in_mem.block.epoch;

        // For convenience in tests, fill-in a header if it's missing.
        // Normally, the header is added in `FinalizeBlock`.
        #[cfg(any(test, feature = "testing"))]
        {
            if self.in_mem.header.is_none() {
                self.in_mem.header = Some(Header {
                    hash: Hash::default(),
                    time: DateTimeUtc::now(),
                    next_validators_hash: Hash::default(),
                });
            }
        }

        let state = BlockStateWrite {
            merkle_tree_stores: self.in_mem.block.tree.stores(),
            header: self.in_mem.header.as_ref(),
            hash: &self.in_mem.block.hash,
            height: self.in_mem.block.height,
            time: self
                .in_mem
                .header
                .as_ref()
                .expect("Must have a block header on commit")
                .time,
            epoch: self.in_mem.block.epoch,
            results: &self.in_mem.block.results,
            pred_epochs: &self.in_mem.block.pred_epochs,
            next_epoch_min_start_height: self
                .in_mem
                .next_epoch_min_start_height,
            next_epoch_min_start_time: self.in_mem.next_epoch_min_start_time,
            update_epoch_blocks_delay: self.in_mem.update_epoch_blocks_delay,
            address_gen: &self.in_mem.address_gen,
            conversion_state: &self.in_mem.conversion_state,
            tx_queue: &self.in_mem.tx_queue,
            ethereum_height: self.in_mem.ethereum_height.as_ref(),
            eth_events_queue: &self.in_mem.eth_events_queue,
        };
        self.db
            .add_block_to_batch(state, &mut batch, is_full_commit)?;
        let header = self
            .in_mem
            .header
            .take()
            .expect("Must have a block header on commit");
        self.in_mem.last_block = Some(LastBlock {
            height: self.in_mem.block.height,
            hash: header.hash.into(),
            time: header.time,
        });
        self.in_mem.last_epoch = self.in_mem.block.epoch;
        if is_full_commit {
            // prune old merkle tree stores
            self.prune_merkle_tree_stores(&mut batch)?;
        }
        self.db.exec_batch(batch)?;
        Ok(())
    }
}

impl<D, H> Deref for FullAccessState<D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    type Target = WlState<D, H>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<D, H> DerefMut for FullAccessState<D, H>
where
    D: DB + for<'iter> DBIter<'iter>,
    H: StorageHasher,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<D, H> WlState<D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    pub fn write_log(&self) -> &WriteLog {
        &self.write_log
    }

    pub fn in_mem(&self) -> &InMemory<H> {
        &self.in_mem
    }

    pub fn in_mem_mut(&mut self) -> &mut InMemory<H> {
        &mut self.in_mem
    }

    pub fn db(&self) -> &D {
        // NOTE: `WlState` must not be allowed mutable access to DB
        &self.db
    }

    pub fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.write_log
    }

    pub fn with_temp_write_log(&self) -> TempWlState<'_, D, H> {
        TempWlState {
            write_log: WriteLog::default(),
            db: &self.db,
            in_mem: &self.in_mem,
        }
    }

    /// Commit the current transaction's write log to the block when it's
    /// accepted by all the triggered validity predicates. Starts a new
    /// transaction write log.
    pub fn commit_tx(&mut self) {
        self.write_log.commit_tx()
    }

    /// Drop the current transaction's write log when it's declined by any of
    /// the triggered validity predicates. Starts a new transaction write log.
    pub fn drop_tx(&mut self) {
        self.write_log.drop_tx()
    }

    /// Delete the provided transaction's hash from storage.
    pub fn delete_tx_hash(&mut self, hash: Hash) -> write_log::Result<()> {
        self.write_log.delete_tx_hash(hash)
    }

    #[inline]
    pub fn get_current_decision_height(&self) -> BlockHeight {
        self.in_mem.get_last_block_height() + 1
    }

    /// Check if we are at a given [`BlockHeight`] offset, `height_offset`,
    /// within the current epoch.
    pub fn is_deciding_offset_within_epoch(&self, height_offset: u64) -> bool {
        let current_decision_height = self.get_current_decision_height();

        let pred_epochs = &self.in_mem.block.pred_epochs;
        let fst_heights_of_each_epoch = pred_epochs.first_block_heights();

        fst_heights_of_each_epoch
            .last()
            .map(|&h| {
                let height_offset_within_epoch = h + height_offset;
                current_decision_height == height_offset_within_epoch
            })
            .unwrap_or(false)
    }

    /// Returns a value from the specified subspace at the given height (or the
    /// last committed height when 0) and the gas cost.
    pub fn db_read_with_height(
        &self,
        key: &storage::Key,
        height: BlockHeight,
    ) -> Result<(Option<Vec<u8>>, u64)> {
        // `0` means last committed height
        if height == BlockHeight(0)
            || height >= self.in_mem().get_last_block_height()
        {
            self.db_read(key)
        } else {
            if !(self.merkle_tree_key_filter)(key) {
                return Ok((None, 0));
            }

            match self.db().read_subspace_val_with_height(
                key,
                height,
                self.in_mem().get_last_block_height(),
            )? {
                Some(v) => {
                    let gas = (key.len() + v.len()) as u64
                        * STORAGE_ACCESS_GAS_PER_BYTE;
                    Ok((Some(v), gas))
                }
                None => {
                    Ok((None, key.len() as u64 * STORAGE_ACCESS_GAS_PER_BYTE))
                }
            }
        }
    }

    /// Write a value to the specified subspace and returns the gas cost and the
    /// size difference
    pub fn db_write(
        &mut self,
        key: &Key,
        value: impl AsRef<[u8]>,
    ) -> Result<(u64, i64)> {
        // Note that this method is the same as `StorageWrite::write_bytes`,
        // but with gas and storage bytes len diff accounting
        tracing::debug!("storage write key {}", key,);
        let value = value.as_ref();
        let is_key_merklized = (self.merkle_tree_key_filter)(key);

        if is_pending_transfer_key(key) {
            // The tree of the bright pool stores the current height for the
            // pending transfer
            let height = self.in_mem.block.height.serialize_to_vec();
            self.in_mem.block.tree.update(key, height)?;
        } else {
            // Update the merkle tree
            if is_key_merklized {
                self.in_mem.block.tree.update(key, value)?;
            }
        }

        let len = value.len();
        let gas = (key.len() + len) as u64 * STORAGE_WRITE_GAS_PER_BYTE;
        let size_diff = self.db.write_subspace_val(
            self.in_mem.block.height,
            key,
            value,
            is_key_merklized,
        )?;
        Ok((gas, size_diff))
    }

    /// Delete the specified subspace and returns the gas cost and the size
    /// difference
    pub fn db_delete(&mut self, key: &Key) -> Result<(u64, i64)> {
        // Note that this method is the same as `StorageWrite::delete`,
        // but with gas and storage bytes len diff accounting
        let mut deleted_bytes_len = 0;
        if self.db_has_key(key)?.0 {
            let is_key_merklized = (self.merkle_tree_key_filter)(key);
            if is_key_merklized {
                self.in_mem.block.tree.delete(key)?;
            }
            deleted_bytes_len = self.db.delete_subspace_val(
                self.in_mem.block.height,
                key,
                is_key_merklized,
            )?;
        }
        let gas = (key.len() + deleted_bytes_len as usize) as u64
            * STORAGE_WRITE_GAS_PER_BYTE;
        Ok((gas, deleted_bytes_len))
    }

    /// Get a Tendermint-compatible existence proof.
    ///
    /// Proofs from the Ethereum bridge pool are not
    /// Tendermint-compatible. Requesting for a key
    /// belonging to the bridge pool will cause this
    /// method to error.
    pub fn get_existence_proof(
        &self,
        key: &Key,
        value: namada_merkle_tree::StorageBytes,
        height: BlockHeight,
    ) -> Result<ProofOps> {
        use std::array;

        // `0` means last committed height
        let height = if height == BlockHeight(0) {
            self.in_mem.get_last_block_height()
        } else {
            height
        };

        if height > self.in_mem.get_last_block_height() {
            if let MembershipProof::ICS23(proof) = self
                .in_mem
                .block
                .tree
                .get_sub_tree_existence_proof(array::from_ref(key), vec![value])
                .map_err(Error::MerkleTreeError)?
            {
                self.in_mem
                    .block
                    .tree
                    .get_sub_tree_proof(key, proof)
                    .map(Into::into)
                    .map_err(Error::MerkleTreeError)
            } else {
                Err(Error::MerkleTreeError(MerkleTreeError::TendermintProof))
            }
        } else {
            let (store_type, _) = StoreType::sub_key(key)?;
            let tree = self.get_merkle_tree(height, Some(store_type))?;
            if let MembershipProof::ICS23(proof) = tree
                .get_sub_tree_existence_proof(array::from_ref(key), vec![value])
                .map_err(Error::MerkleTreeError)?
            {
                tree.get_sub_tree_proof(key, proof)
                    .map(Into::into)
                    .map_err(Error::MerkleTreeError)
            } else {
                Err(Error::MerkleTreeError(MerkleTreeError::TendermintProof))
            }
        }
    }

    /// Get the non-existence proof
    pub fn get_non_existence_proof(
        &self,
        key: &Key,
        height: BlockHeight,
    ) -> Result<ProofOps> {
        // `0` means last committed height
        let height = if height == BlockHeight(0) {
            self.in_mem.get_last_block_height()
        } else {
            height
        };

        if height > self.in_mem.get_last_block_height() {
            Err(Error::Temporary {
                error: format!(
                    "The block at the height {} hasn't committed yet",
                    height,
                ),
            })
        } else {
            let (store_type, _) = StoreType::sub_key(key)?;
            self.get_merkle_tree(height, Some(store_type))?
                .get_non_existence_proof(key)
                .map(Into::into)
                .map_err(Error::MerkleTreeError)
        }
    }

    /// Rebuild Merkle tree with diffs in the DB.
    /// Base tree and the specified `store_type` subtree is rebuilt.
    /// If `store_type` isn't given, full Merkle tree is restored.
    pub fn get_merkle_tree(
        &self,
        height: BlockHeight,
        store_type: Option<StoreType>,
    ) -> Result<MerkleTree<H>> {
        // `0` means last committed height
        let height = if height == BlockHeight(0) {
            self.in_mem.get_last_block_height()
        } else {
            height
        };

        let epoch = self
            .in_mem
            .block
            .pred_epochs
            .get_epoch(height)
            .unwrap_or(Epoch::default());
        let epoch_start_height = match self
            .in_mem
            .block
            .pred_epochs
            .get_start_height_of_epoch(epoch)
        {
            Some(height) if height == BlockHeight(0) => BlockHeight(1),
            Some(height) => height,
            None => BlockHeight(1),
        };
        let stores = self
            .db
            .read_merkle_tree_stores(epoch, epoch_start_height, store_type)?
            .ok_or(Error::NoMerkleTree { height })?;
        let prefix = store_type.and_then(|st| st.provable_prefix());
        let mut tree = match store_type {
            Some(_) => MerkleTree::<H>::new_partial(stores),
            None => MerkleTree::<H>::new(stores).expect("invalid stores"),
        };
        // Restore the tree state with diffs
        let mut target_height = epoch_start_height;
        while target_height < height {
            target_height = target_height.next_height();
            let mut old_diff_iter =
                self.db.iter_old_diffs(target_height, prefix.as_ref());
            let mut new_diff_iter =
                self.db.iter_new_diffs(target_height, prefix.as_ref());

            let mut old_diff = old_diff_iter.next();
            let mut new_diff = new_diff_iter.next();
            loop {
                match (&old_diff, &new_diff) {
                    (Some(old), Some(new)) => {
                        let old_key = Key::parse(old.0.clone())
                            .expect("the key should be parsable");
                        let new_key = Key::parse(new.0.clone())
                            .expect("the key should be parsable");

                        // compare keys as String
                        match old.0.cmp(&new.0) {
                            Ordering::Equal => {
                                // the value was updated
                                if (self.merkle_tree_key_filter)(&new_key) {
                                    tree.update(
                                        &new_key,
                                        if is_pending_transfer_key(&new_key) {
                                            target_height.serialize_to_vec()
                                        } else {
                                            new.1.clone()
                                        },
                                    )?;
                                }
                                old_diff = old_diff_iter.next();
                                new_diff = new_diff_iter.next();
                            }
                            Ordering::Less => {
                                // the value was deleted
                                if (self.merkle_tree_key_filter)(&old_key) {
                                    tree.delete(&old_key)?;
                                }
                                old_diff = old_diff_iter.next();
                            }
                            Ordering::Greater => {
                                // the value was inserted
                                if (self.merkle_tree_key_filter)(&new_key) {
                                    tree.update(
                                        &new_key,
                                        if is_pending_transfer_key(&new_key) {
                                            target_height.serialize_to_vec()
                                        } else {
                                            new.1.clone()
                                        },
                                    )?;
                                }
                                new_diff = new_diff_iter.next();
                            }
                        }
                    }
                    (Some(old), None) => {
                        // the value was deleted
                        let key = Key::parse(old.0.clone())
                            .expect("the key should be parsable");

                        if (self.merkle_tree_key_filter)(&key) {
                            tree.delete(&key)?;
                        }

                        old_diff = old_diff_iter.next();
                    }
                    (None, Some(new)) => {
                        // the value was inserted
                        let key = Key::parse(new.0.clone())
                            .expect("the key should be parsable");

                        if (self.merkle_tree_key_filter)(&key) {
                            tree.update(
                                &key,
                                if is_pending_transfer_key(&key) {
                                    target_height.serialize_to_vec()
                                } else {
                                    new.1.clone()
                                },
                            )?;
                        }

                        new_diff = new_diff_iter.next();
                    }
                    (None, None) => break,
                }
            }
        }
        if let Some(st) = store_type {
            // Add the base tree with the given height
            let mut stores = self
                .db
                .read_merkle_tree_stores(epoch, height, Some(StoreType::Base))?
                .ok_or(Error::NoMerkleTree { height })?;
            let restored_stores = tree.stores();
            // Set the root and store of the rebuilt subtree
            stores.set_root(&st, *restored_stores.root(&st));
            stores.set_store(restored_stores.store(&st).to_owned());
            tree = MerkleTree::<H>::new_partial(stores);
        }
        Ok(tree)
    }

    /// Get the timestamp of the last committed block, or the current timestamp
    /// if no blocks have been produced yet
    pub fn get_last_block_timestamp(&self) -> Result<DateTimeUtc> {
        let last_block_height = self.in_mem.get_block_height().0;

        Ok(self
            .db
            .read_block_header(last_block_height)?
            .map_or_else(DateTimeUtc::now, |header| header.time))
    }
}

impl<D, H> TempWlState<'_, D, H>
where
    D: 'static + DB + for<'iter> DBIter<'iter>,
    H: 'static + StorageHasher,
{
    pub fn write_log(&self) -> &WriteLog {
        &self.write_log
    }

    pub fn in_mem(&self) -> &InMemory<H> {
        self.in_mem
    }

    pub fn db(&self) -> &D {
        self.db
    }

    pub fn write_log_mut(&mut self) -> &mut WriteLog {
        &mut self.write_log
    }

    /// Check if the given tx hash has already been processed
    pub fn has_replay_protection_entry(&self, hash: &Hash) -> Result<bool> {
        if let Some(present) = self.write_log.has_replay_protection_entry(hash)
        {
            return Ok(present);
        }

        self.db()
            .has_replay_protection_entry(hash)
            .map_err(Error::DbError)
    }

    /// Check if the given tx hash has already been committed to storage
    pub fn has_committed_replay_protection_entry(
        &self,
        hash: &Hash,
    ) -> Result<bool> {
        self.db()
            .has_replay_protection_entry(hash)
            .map_err(Error::DbError)
    }
}

#[macro_export]
macro_rules! impl_storage_read {
    ($($type:ty)*) => {
        impl<D, H> StorageRead for $($type)*
        where
            D: 'static + DB + for<'iter> DBIter<'iter>,
            H: 'static + StorageHasher,
        {
            type PrefixIter<'iter> = PrefixIter<'iter, D> where Self: 'iter;

            fn read_bytes(
                &self,
                key: &storage::Key,
            ) -> namada_storage::Result<Option<Vec<u8>>> {
                // try to read from the write log first
                let (log_val, gas) = self.write_log().read(key);
                self.charge_gas(gas).into_storage_result()?;
                match log_val {
                    Some(write_log::StorageModification::Write { ref value }) => {
                        Ok(Some(value.clone()))
                    }
                    Some(write_log::StorageModification::Delete) => Ok(None),
                    Some(write_log::StorageModification::InitAccount {
                        ref vp_code_hash,
                    }) => Ok(Some(vp_code_hash.to_vec())),
                    Some(write_log::StorageModification::Temp { ref value }) => {
                        Ok(Some(value.clone()))
                    }
                    None => {
                        // when not found in write log, try to read from the storage
                        let (value, gas) = self.db_read(key).into_storage_result()?;
                        dbg!(key, value.is_some());
                        self.charge_gas(gas).into_storage_result()?;
                        Ok(value)
                    }
                }
            }

            fn has_key(&self, key: &storage::Key) -> namada_storage::Result<bool> {
                // try to read from the write log first
                let (log_val, gas) = self.write_log().read(key);
                self.charge_gas(gas).into_storage_result()?;
                match log_val {
                    Some(&write_log::StorageModification::Write { .. })
                    | Some(&write_log::StorageModification::InitAccount { .. })
                    | Some(&write_log::StorageModification::Temp { .. }) => Ok(true),
                    Some(&write_log::StorageModification::Delete) => {
                        // the given key has been deleted
                        Ok(false)
                    }
                    None => {
                        // when not found in write log, try to check the storage
                        let (present, gas) = self.db_has_key(key).into_storage_result()?;
                        self.charge_gas(gas).into_storage_result()?;
                        Ok(present)
                    }
                }
            }

            fn iter_prefix<'iter>(
                &'iter self,
                prefix: &storage::Key,
            ) -> namada_storage::Result<Self::PrefixIter<'iter>> {
                let (iter, gas) =
                    iter_prefix_post(self.write_log(), self.db(), prefix);
                self.charge_gas(gas).into_storage_result()?;
                Ok(iter)
            }

            fn iter_next<'iter>(
                &'iter self,
                iter: &mut Self::PrefixIter<'iter>,
            ) -> namada_storage::Result<Option<(String, Vec<u8>)>> {
                iter.next().map(|(key, val, gas)| {
                    self.charge_gas(gas).into_storage_result()?;
                    Ok((key, val))
                }).transpose()
            }

            fn get_chain_id(
                &self,
            ) -> std::result::Result<String, namada_storage::Error> {
                let (chain_id, gas) = self.in_mem().get_chain_id();
                self.charge_gas(gas).into_storage_result()?;
                Ok(chain_id)
            }

            fn get_block_height(
                &self,
            ) -> std::result::Result<storage::BlockHeight, namada_storage::Error> {
                let (height, gas) = self.in_mem().get_block_height();
                self.charge_gas(gas).into_storage_result()?;
                Ok(height)
            }

            fn get_block_header(
                &self,
                height: storage::BlockHeight,
            ) -> std::result::Result<Option<storage::Header>, namada_storage::Error>
            {
                let (header, gas) =
                    StateRead::get_block_header(self, Some(height)).into_storage_result()?;
                self.charge_gas(gas).into_storage_result()?;
                Ok(header)
            }

            fn get_block_hash(
                &self,
            ) -> std::result::Result<storage::BlockHash, namada_storage::Error> {
                let (hash, gas) = self.in_mem().get_block_hash();
                self.charge_gas(gas).into_storage_result()?;
                Ok(hash)
            }

            fn get_block_epoch(
                &self,
            ) -> std::result::Result<storage::Epoch, namada_storage::Error> {
                let (epoch, gas) = self.in_mem().get_current_epoch();
                self.charge_gas(gas).into_storage_result()?;
                Ok(epoch)
            }

            fn get_pred_epochs(&self) -> namada_storage::Result<Epochs> {
                self.charge_gas(
                    namada_gas::STORAGE_ACCESS_GAS_PER_BYTE,
                ).into_storage_result()?;
                Ok(self.in_mem().block.pred_epochs.clone())
            }

            fn get_tx_index(
                &self,
            ) -> std::result::Result<storage::TxIndex, namada_storage::Error> {
                self.charge_gas(
                    namada_gas::STORAGE_ACCESS_GAS_PER_BYTE,
                ).into_storage_result()?;
                Ok(self.in_mem().tx_index)
            }

            fn get_native_token(&self) -> namada_storage::Result<Address> {
                self.charge_gas(
                    namada_gas::STORAGE_ACCESS_GAS_PER_BYTE,
                ).into_storage_result()?;
                Ok(self.in_mem().native_token.clone())
            }
        }
    }
}

#[macro_export]
macro_rules! impl_storage_write {
    ($($type:ty)*) => {
        impl<D, H> StorageWrite for $($type)*
        where
            D: 'static + DB + for<'iter> DBIter<'iter>,
            H: 'static + StorageHasher,
        {
            fn write_bytes(
                &mut self,
                key: &storage::Key,
                val: impl AsRef<[u8]>,
            ) -> namada_storage::Result<()> {
                let (gas, _size_diff) = self
                    .write_log_mut()
                    .write(key, val.as_ref().to_vec())
                    .into_storage_result()?;
                self.charge_gas(gas).into_storage_result()?;
                Ok(())
            }

            fn delete(&mut self, key: &storage::Key) -> namada_storage::Result<()> {
                let (gas, _size_diff) = self
                    .write_log_mut()
                    .delete(key)
                    .into_storage_result()?;
                self.charge_gas(gas).into_storage_result()?;
                Ok(())
            }
        }
    };
}

// Note: `FullAccessState` writes to a write-log at block-level, while all the
// other `StorageWrite` impls write at tx-level.
macro_rules! impl_storage_write_by_protocol {
    ($($type:ty)*) => {
        impl<D, H> StorageWrite for $($type)*
        where
            D: 'static + DB + for<'iter> DBIter<'iter>,
            H: 'static + StorageHasher,
        {
            fn write_bytes(
                &mut self,
                key: &storage::Key,
                val: impl AsRef<[u8]>,
            ) -> namada_storage::Result<()> {
                self
                    .write_log_mut()
                    .protocol_write(key, val.as_ref().to_vec())
                    .into_storage_result()?;
                Ok(())
            }

            fn delete(&mut self, key: &storage::Key) -> namada_storage::Result<()> {
                self
                    .write_log_mut()
                    .protocol_delete(key)
                    .into_storage_result()?;
                Ok(())
            }
        }
    };
}

impl_storage_read!(FullAccessState<D, H>);
impl_storage_read!(WlState<D, H>);
impl_storage_read!(TempWlState<'_, D, H>);
impl_storage_write_by_protocol!(FullAccessState<D, H>);
impl_storage_write_by_protocol!(WlState<D, H>);
impl_storage_write_by_protocol!(TempWlState<'_, D, H>);

impl_storage_read!(TxHostEnvState<'_, D, H>);
impl_storage_read!(VpHostEnvState<'_, D, H>);
impl_storage_write!(TxHostEnvState<'_, D, H>);

/// The ledger's state
#[derive(Debug)]
pub struct InMemory<H>
where
    H: StorageHasher,
{
    /// The ID of the chain
    pub chain_id: ChainId,
    /// The address of the native token - this is not stored in DB, but read
    /// from genesis
    pub native_token: Address,
    /// Block storage data
    pub block: BlockStorage<H>,
    /// During `FinalizeBlock`, this is the header of the block that is
    /// going to be committed. After a block is committed, this is reset to
    /// `None` until the next `FinalizeBlock` phase is reached.
    pub header: Option<Header>,
    /// The most recently committed block, if any.
    pub last_block: Option<LastBlock>,
    /// The epoch of the most recently committed block. If it is `Epoch(0)`,
    /// then no block may have been committed for this chain yet.
    pub last_epoch: Epoch,
    /// Minimum block height at which the next epoch may start
    pub next_epoch_min_start_height: BlockHeight,
    /// Minimum block time at which the next epoch may start
    pub next_epoch_min_start_time: DateTimeUtc,
    /// The current established address generator
    pub address_gen: EstablishedAddressGen,
    /// We delay the switch to a new epoch by the number of blocks set in here.
    /// This is `Some` when minimum number of blocks has been created and
    /// minimum time has passed since the beginning of the last epoch.
    /// Once the value is `Some(0)`, we're ready to switch to a new epoch and
    /// this is reset back to `None`.
    pub update_epoch_blocks_delay: Option<u32>,
    /// The shielded transaction index
    pub tx_index: TxIndex,
    /// The currently saved conversion state
    pub conversion_state: ConversionState,
    /// Wrapper txs to be decrypted in the next block proposal
    pub tx_queue: TxQueue,
    /// Queue of expired transactions that need to be retransmitted.
    ///
    /// These transactions do not need to be persisted, as they are
    /// retransmitted at the **COMMIT** phase immediately following
    /// the block when they were queued.
    pub expired_txs_queue: ExpiredTxsQueue,
    /// The latest block height on Ethereum processed, if
    /// the bridge is enabled.
    pub ethereum_height: Option<ethereum_structs::BlockHeight>,
    /// The queue of Ethereum events to be processed in order.
    pub eth_events_queue: EthEventsQueue,
    /// How many block heights in the past can the storage be queried
    pub storage_read_past_height_limit: Option<u64>,
}

/// Last committed block
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct LastBlock {
    /// Block height
    pub height: BlockHeight,
    /// Block hash
    pub hash: BlockHash,
    /// Block time
    pub time: DateTimeUtc,
}

/// The block storage data
#[derive(Debug)]
pub struct BlockStorage<H: StorageHasher> {
    /// Merkle tree of all the other data in block storage
    pub tree: MerkleTree<H>,
    /// During `FinalizeBlock`, this is updated to be the hash of the block
    /// that is going to be committed. If it is `BlockHash::default()`,
    /// then no `FinalizeBlock` stage has been reached yet.
    pub hash: BlockHash,
    /// From the start of `FinalizeBlock` until the end of `Commit`, this is
    /// height of the block that is going to be committed. Otherwise, it is the
    /// height of the most recently committed block, or `BlockHeight::sentinel`
    /// (0) if no block has been committed yet.
    pub height: BlockHeight,
    /// From the start of `FinalizeBlock` until the end of `Commit`, this is
    /// height of the block that is going to be committed. Otherwise it is the
    /// epoch of the most recently committed block, or `Epoch(0)` if no block
    /// has been committed yet.
    pub epoch: Epoch,
    /// Results of applying transactions
    pub results: BlockResults,
    /// Predecessor block epochs
    pub pred_epochs: Epochs,
}

pub fn merklize_all_keys(_key: &storage::Key) -> bool {
    true
}

#[allow(missing_docs)]
#[derive(Error, Debug)]
pub enum Error {
    #[error("TEMPORARY error: {error}")]
    Temporary { error: String },
    #[error("Found an unknown key: {key}")]
    UnknownKey { key: String },
    #[error("Storage key error {0}")]
    KeyError(namada_core::storage::Error),
    #[error("Coding error: {0}")]
    CodingError(#[from] namada_core::DecodeError),
    #[error("Merkle tree error: {0}")]
    MerkleTreeError(MerkleTreeError),
    #[error("DB error: {0}")]
    DBError(String),
    #[error("Borsh (de)-serialization error: {0}")]
    BorshCodingError(std::io::Error),
    #[error("Merkle tree at the height {height} is not stored")]
    NoMerkleTree { height: BlockHeight },
    #[error("Code hash error: {0}")]
    InvalidCodeHash(HashError),
    #[error("DB error: {0}")]
    DbError(#[from] namada_storage::DbError),
    #[error("{0}")]
    Gas(namada_gas::Error),
    #[error("{0}")]
    StorageError(#[from] namada_storage::Error),
}

impl<H> InMemory<H>
where
    H: StorageHasher,
{
    /// Create a new instance of the state
    pub fn new(
        chain_id: ChainId,
        native_token: Address,
        storage_read_past_height_limit: Option<u64>,
    ) -> Self {
        let block = BlockStorage {
            tree: MerkleTree::default(),
            hash: BlockHash::default(),
            height: BlockHeight::default(),
            epoch: Epoch::default(),
            pred_epochs: Epochs::default(),
            results: BlockResults::default(),
        };
        InMemory::<H> {
            chain_id,
            block,
            header: None,
            last_block: None,
            last_epoch: Epoch::default(),
            next_epoch_min_start_height: BlockHeight::default(),
            next_epoch_min_start_time: DateTimeUtc::now(),
            address_gen: EstablishedAddressGen::new(
                "Privacy is a function of liberty.",
            ),
            update_epoch_blocks_delay: None,
            tx_index: TxIndex::default(),
            conversion_state: ConversionState::default(),
            tx_queue: TxQueue::default(),
            expired_txs_queue: ExpiredTxsQueue::default(),
            native_token,
            ethereum_height: None,
            eth_events_queue: EthEventsQueue::default(),
            storage_read_past_height_limit,
        }
    }

    /// Returns the Merkle root hash and the height of the committed block. If
    /// no block exists, returns None.
    pub fn get_state(&self) -> Option<(MerkleRoot, u64)> {
        if self.block.height.0 != 0 {
            Some((self.block.tree.root(), self.block.height.0))
        } else {
            None
        }
    }

    /// Find the root hash of the merkle tree
    pub fn merkle_root(&self) -> MerkleRoot {
        self.block.tree.root()
    }

    /// Set the block header.
    /// The header is not in the Merkle tree as it's tracked by Tendermint.
    /// Hence, we don't update the tree when this is set.
    pub fn set_header(&mut self, header: Header) -> Result<()> {
        self.header = Some(header);
        Ok(())
    }

    /// Block data is in the Merkle tree as it's tracked by Tendermint in the
    /// block header. Hence, we don't update the tree when this is set.
    pub fn begin_block(
        &mut self,
        hash: BlockHash,
        height: BlockHeight,
    ) -> Result<()> {
        self.block.hash = hash;
        self.block.height = height;
        Ok(())
    }

    /// Get the chain ID as a raw string
    pub fn get_chain_id(&self) -> (String, u64) {
        (
            self.chain_id.to_string(),
            CHAIN_ID_LENGTH as u64 * MEMORY_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Get the block height
    pub fn get_block_height(&self) -> (BlockHeight, u64) {
        (
            self.block.height,
            BLOCK_HEIGHT_LENGTH as u64 * MEMORY_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Get the block hash
    pub fn get_block_hash(&self) -> (BlockHash, u64) {
        (
            self.block.hash.clone(),
            BLOCK_HASH_LENGTH as u64 * MEMORY_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Get the current (yet to be committed) block epoch
    pub fn get_current_epoch(&self) -> (Epoch, u64) {
        (
            self.block.epoch,
            EPOCH_TYPE_LENGTH as u64 * MEMORY_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Get the epoch of the last committed block
    pub fn get_last_epoch(&self) -> (Epoch, u64) {
        (
            self.last_epoch,
            EPOCH_TYPE_LENGTH as u64 * MEMORY_ACCESS_GAS_PER_BYTE,
        )
    }

    /// Initialize the first epoch. The first epoch begins at genesis time.
    pub fn init_genesis_epoch(
        &mut self,
        initial_height: BlockHeight,
        genesis_time: DateTimeUtc,
        parameters: &Parameters,
    ) -> Result<()> {
        let EpochDuration {
            min_num_of_blocks,
            min_duration,
        } = parameters.epoch_duration;
        self.next_epoch_min_start_height = initial_height + min_num_of_blocks;
        self.next_epoch_min_start_time = genesis_time + min_duration;
        self.block.pred_epochs = Epochs {
            first_block_heights: vec![initial_height],
        };
        self.update_epoch_in_merkle_tree()
    }

    /// Get the current conversions
    pub fn get_conversion_state(&self) -> &ConversionState {
        &self.conversion_state
    }

    /// Update the merkle tree with epoch data
    fn update_epoch_in_merkle_tree(&mut self) -> Result<()> {
        let key_prefix: Key =
            Address::Internal(InternalAddress::PoS).to_db_key().into();

        let key = key_prefix
            .push(&"epoch_start_height".to_string())
            .map_err(Error::KeyError)?;
        self.block
            .tree
            .update(&key, encode(&self.next_epoch_min_start_height))?;

        let key = key_prefix
            .push(&"epoch_start_time".to_string())
            .map_err(Error::KeyError)?;
        self.block
            .tree
            .update(&key, encode(&self.next_epoch_min_start_time))?;

        let key = key_prefix
            .push(&"current_epoch".to_string())
            .map_err(Error::KeyError)?;
        self.block.tree.update(&key, encode(&self.block.epoch))?;

        Ok(())
    }

    /// Get the height of the last committed block or 0 if no block has been
    /// committed yet. The first block is at height 1.
    pub fn get_last_block_height(&self) -> BlockHeight {
        self.last_block
            .as_ref()
            .map(|b| b.height)
            .unwrap_or_default()
    }

    /// Get the oldest epoch where we can read a value
    pub fn get_oldest_epoch(&self) -> Epoch {
        let oldest_height = match self.storage_read_past_height_limit {
            Some(limit) if limit < self.get_last_block_height().0 => {
                (self.get_last_block_height().0 - limit).into()
            }
            _ => BlockHeight(1),
        };
        self.block
            .pred_epochs
            .get_epoch(oldest_height)
            .unwrap_or_default()
    }
}

impl From<MerkleTreeError> for Error {
    fn from(error: MerkleTreeError) -> Self {
        Self::MerkleTreeError(error)
    }
}

/// Prefix iterator for [`StorageRead`] implementations.
#[derive(Debug)]
pub struct PrefixIter<'iter, D>
where
    D: DB + DBIter<'iter>,
{
    /// Peekable storage iterator
    pub storage_iter: Peekable<<D as DBIter<'iter>>::PrefixIter>,
    /// Peekable write log iterator
    pub write_log_iter: Peekable<write_log::PrefixIter>,
}

/// Iterate write-log storage items prior to a tx execution, matching the
/// given prefix. Returns the iterator and gas cost.
pub fn iter_prefix_pre<'a, D>(
    // We cannot use e.g. `&'a State`, because it doesn't live long
    // enough - the lifetime of the `PrefixIter` must depend on the lifetime of
    // references to the `WriteLog` and `DB`.
    write_log: &'a WriteLog,
    db: &'a D,
    prefix: &storage::Key,
) -> (PrefixIter<'a, D>, u64)
where
    D: DB + for<'iter> DBIter<'iter>,
{
    let storage_iter = db.iter_prefix(Some(prefix)).peekable();
    let write_log_iter = write_log.iter_prefix_pre(prefix).peekable();
    (
        PrefixIter::<D> {
            storage_iter,
            write_log_iter,
        },
        prefix.len() as u64 * namada_gas::STORAGE_ACCESS_GAS_PER_BYTE,
    )
}

/// Iterate write-log storage items posterior to a tx execution, matching the
/// given prefix. Returns the iterator and gas cost.
pub fn iter_prefix_post<'a, D>(
    // We cannot use e.g. `&'a State`, because it doesn't live long
    // enough - the lifetime of the `PrefixIter` must depend on the lifetime of
    // references to the `WriteLog` and `DB`.
    write_log: &'a WriteLog,
    db: &'a D,
    prefix: &storage::Key,
) -> (PrefixIter<'a, D>, u64)
where
    D: DB + for<'iter> DBIter<'iter>,
{
    let storage_iter = db.iter_prefix(Some(prefix)).peekable();
    let write_log_iter = write_log.iter_prefix_post(prefix).peekable();
    (
        PrefixIter::<D> {
            storage_iter,
            write_log_iter,
        },
        prefix.len() as u64 * namada_gas::STORAGE_ACCESS_GAS_PER_BYTE,
    )
}

impl<'iter, D> Iterator for PrefixIter<'iter, D>
where
    D: DB + DBIter<'iter>,
{
    type Item = (String, Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        enum Next {
            ReturnWl { advance_storage: bool },
            ReturnStorage,
        }
        loop {
            let what: Next;
            {
                let storage_peeked = self.storage_iter.peek();
                let wl_peeked = self.write_log_iter.peek();
                match (storage_peeked, wl_peeked) {
                    (None, None) => return None,
                    (None, Some(_)) => {
                        what = Next::ReturnWl {
                            advance_storage: false,
                        };
                    }
                    (Some(_), None) => {
                        what = Next::ReturnStorage;
                    }
                    (Some((storage_key, _, _)), Some((wl_key, _))) => {
                        if wl_key <= storage_key {
                            what = Next::ReturnWl {
                                advance_storage: wl_key == storage_key,
                            };
                        } else {
                            what = Next::ReturnStorage;
                        }
                    }
                }
            }
            match what {
                Next::ReturnWl { advance_storage } => {
                    if advance_storage {
                        let _ = self.storage_iter.next();
                    }

                    if let Some((key, modification)) =
                        self.write_log_iter.next()
                    {
                        match modification {
                            write_log::StorageModification::Write { value }
                            | write_log::StorageModification::Temp { value } => {
                                let gas = value.len() as u64;
                                return Some((key, value, gas));
                            }
                            write_log::StorageModification::InitAccount {
                                vp_code_hash,
                            } => {
                                let gas = vp_code_hash.len() as u64;
                                return Some((key, vp_code_hash.to_vec(), gas));
                            }
                            write_log::StorageModification::Delete => {
                                continue;
                            }
                        }
                    }
                }
                Next::ReturnStorage => {
                    if let Some(next) = self.storage_iter.next() {
                        return Some(next);
                    }
                }
            }
        }
    }
}

/// Helpers for testing components that depend on storage
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use namada_core::address;
    use namada_core::hash::Sha256Hasher;

    use super::mockdb::MockDB;
    use super::*;

    pub type TestState = FullAccessState<MockDB, Sha256Hasher>;

    impl Default for TestState {
        fn default() -> Self {
            Self(WlState {
                write_log: Default::default(),
                db: MockDB::default(),
                in_mem: Default::default(),
                merkle_tree_key_filter: merklize_all_keys,
            })
        }
    }

    /// In memory State for testing.
    pub type InMemoryState = InMemory<Sha256Hasher>;

    impl Default for InMemoryState {
        fn default() -> Self {
            let chain_id = ChainId::default();
            let tree = MerkleTree::default();
            let block = BlockStorage {
                tree,
                hash: BlockHash::default(),
                height: BlockHeight::default(),
                epoch: Epoch::default(),
                pred_epochs: Epochs::default(),
                results: BlockResults::default(),
            };
            Self {
                chain_id,
                block,
                header: None,
                last_block: None,
                last_epoch: Epoch::default(),
                next_epoch_min_start_height: BlockHeight::default(),
                next_epoch_min_start_time: DateTimeUtc::now(),
                address_gen: EstablishedAddressGen::new(
                    "Test address generator seed",
                ),
                update_epoch_blocks_delay: None,
                tx_index: TxIndex::default(),
                conversion_state: ConversionState::default(),
                tx_queue: TxQueue::default(),
                expired_txs_queue: ExpiredTxsQueue::default(),
                native_token: address::testing::nam(),
                ethereum_height: None,
                eth_events_queue: EthEventsQueue::default(),
                storage_read_past_height_limit: Some(1000),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::{TimeZone, Utc};
    use namada_core::dec::Dec;
    use namada_core::time::{self, Duration};
    use namada_core::token;
    use namada_parameters::Parameters;
    use proptest::prelude::*;
    use proptest::test_runner::Config;

    use super::testing::*;
    use super::*;

    prop_compose! {
        /// Setup test input data with arbitrary epoch duration, epoch start
        /// height and time, and a block height and time that are greater than
        /// the epoch start height and time, and the change to be applied to
        /// the epoch duration parameters.
        fn arb_and_epoch_duration_start_and_block()
        (
            start_height in 0..1000_u64,
            start_time in 0..10000_i64,
            min_num_of_blocks in 1..10_u64,
            min_duration in 1..100_i64,
            max_expected_time_per_block in 1..100_i64,
        )
        (
            min_num_of_blocks in Just(min_num_of_blocks),
            min_duration in Just(min_duration),
            max_expected_time_per_block in Just(max_expected_time_per_block),
            start_height in Just(start_height),
            start_time in Just(start_time),
            block_height in start_height + 1..(start_height + 2 * min_num_of_blocks),
            block_time in start_time + 1..(start_time + 2 * min_duration),
            // Delta will be applied on the `min_num_of_blocks` parameter
            min_blocks_delta in -(min_num_of_blocks as i64 - 1)..5,
            // Delta will be applied on the `min_duration` parameter
            min_duration_delta in -(min_duration - 1)..50,
            // Delta will be applied on the `max_expected_time_per_block` parameter
            max_time_per_block_delta in -(max_expected_time_per_block - 1)..50,
        ) -> (EpochDuration, i64, BlockHeight, DateTimeUtc, BlockHeight, DateTimeUtc,
                i64, i64, i64) {
            let epoch_duration = EpochDuration {
                min_num_of_blocks,
                min_duration: Duration::seconds(min_duration).into(),
            };
            (epoch_duration, max_expected_time_per_block,
                BlockHeight(start_height), Utc.timestamp_opt(start_time, 0).single().expect("expected valid timestamp").into(),
                BlockHeight(block_height), Utc.timestamp_opt(block_time, 0).single().expect("expected valid timestamp").into(),
                min_blocks_delta, min_duration_delta, max_time_per_block_delta)
        }
    }

    proptest! {
        #![proptest_config(Config {
            cases: 10,
            .. Config::default()
        })]
        /// Test that:
        /// 1. When the minimum blocks have been created since the epoch
        ///    start height and minimum time passed since the epoch start time,
        ///    a new epoch must start.
        /// 2. When the epoch duration parameters change, the current epoch's
        ///    duration doesn't change, but the next one does.
        #[test]
        fn update_epoch_after_its_duration(
            (epoch_duration, max_expected_time_per_block, start_height, start_time, block_height, block_time,
            min_blocks_delta, min_duration_delta, max_time_per_block_delta)
            in arb_and_epoch_duration_start_and_block())
        {
            let mut state =TestState::default();
            state.in_mem_mut().next_epoch_min_start_height=
                        start_height + epoch_duration.min_num_of_blocks;
            state.in_mem_mut().next_epoch_min_start_time=
                        start_time + epoch_duration.min_duration;
            let mut parameters = Parameters {
                max_tx_bytes: 1024 * 1024,
                max_proposal_bytes: Default::default(),
                max_block_gas: 20_000_000,
                epoch_duration: epoch_duration.clone(),
                max_expected_time_per_block: Duration::seconds(max_expected_time_per_block).into(),
                vp_allowlist: vec![],
                tx_allowlist: vec![],
                implicit_vp_code_hash: Some(Hash::zero()),
                epochs_per_year: 100,
                max_signatures_per_transaction: 15,
                staked_ratio: Dec::new(1,1).expect("Cannot fail"),
                pos_inflation_amount: token::Amount::zero(),
                fee_unshielding_gas_limit: 20_000,
                fee_unshielding_descriptions_limit: 15,
                minimum_gas_price: BTreeMap::default(),
            };
            namada_parameters::init_storage(&parameters, &mut state).unwrap();
            // Initialize pred_epochs to the current height
            let height = state.in_mem().block.height;
            state
                .in_mem_mut()
                .block
                .pred_epochs
                .new_epoch(height);

            let epoch_before = state.in_mem().last_epoch;
            assert_eq!(epoch_before, state.in_mem().block.epoch);

            // Try to apply the epoch update
            state.update_epoch(block_height, block_time).unwrap();

            // Test for 1.
            if block_height.0 - start_height.0
                >= epoch_duration.min_num_of_blocks
                && time::duration_passed(
                    block_time,
                    start_time,
                    epoch_duration.min_duration,
                )
            {
                // Update will now be enqueued for 2 blocks in the future
                assert_eq!(state.in_mem().block.epoch, epoch_before);
                assert_eq!(state.in_mem().update_epoch_blocks_delay, Some(2));

                let block_height = block_height + 1;
                let block_time = block_time + Duration::seconds(1);
                state.update_epoch(block_height, block_time).unwrap();
                assert_eq!(state.in_mem().block.epoch, epoch_before);
                assert_eq!(state.in_mem().update_epoch_blocks_delay, Some(1));

                let block_height = block_height + 1;
                let block_time = block_time + Duration::seconds(1);
                state.update_epoch(block_height, block_time).unwrap();
                assert_eq!(state.in_mem().block.epoch, epoch_before.next());
                assert!(state.in_mem().update_epoch_blocks_delay.is_none());

                assert_eq!(state.in_mem().next_epoch_min_start_height,
                    block_height + epoch_duration.min_num_of_blocks);
                assert_eq!(state.in_mem().next_epoch_min_start_time,
                    block_time + epoch_duration.min_duration);
                assert_eq!(
                    state.in_mem().block.pred_epochs.get_epoch(BlockHeight(block_height.0 - 1)),
                    Some(epoch_before));
                assert_eq!(
                    state.in_mem().block.pred_epochs.get_epoch(block_height),
                    Some(epoch_before.next()));
            } else {
                assert!(state.in_mem().update_epoch_blocks_delay.is_none());
                assert_eq!(state.in_mem().block.epoch, epoch_before);
                assert_eq!(
                    state.in_mem().block.pred_epochs.get_epoch(BlockHeight(block_height.0 - 1)),
                    Some(epoch_before));
                assert_eq!(
                    state.in_mem().block.pred_epochs.get_epoch(block_height),
                    Some(epoch_before));
            }
            // Last epoch should only change when the block is committed
            assert_eq!(state.in_mem().last_epoch, epoch_before);

            // Update the epoch duration parameters
            parameters.epoch_duration.min_num_of_blocks =
                (parameters.epoch_duration.min_num_of_blocks as i64 + min_blocks_delta) as u64;
            let min_duration: i64 = parameters.epoch_duration.min_duration.0 as _;
            parameters.epoch_duration.min_duration =
                Duration::seconds(min_duration + min_duration_delta).into();
            parameters.max_expected_time_per_block =
                Duration::seconds(max_expected_time_per_block + max_time_per_block_delta).into();
            namada_parameters::update_max_expected_time_per_block_parameter(&mut state, &parameters.max_expected_time_per_block).unwrap();
            namada_parameters::update_epoch_parameter(&mut state, &parameters.epoch_duration).unwrap();

            // Test for 2.
            let epoch_before = state.in_mem().block.epoch;
            let height_of_update = state.in_mem().next_epoch_min_start_height.0 ;
            let time_of_update = state.in_mem().next_epoch_min_start_time;
            let height_before_update = BlockHeight(height_of_update - 1);
            let height_of_update = BlockHeight(height_of_update);
            let time_before_update = time_of_update - Duration::seconds(1);

            // No update should happen before both epoch duration conditions are
            // satisfied
            state.update_epoch(height_before_update, time_before_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before);
            assert!(state.in_mem().update_epoch_blocks_delay.is_none());
            state.update_epoch(height_of_update, time_before_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before);
            assert!(state.in_mem().update_epoch_blocks_delay.is_none());
            state.update_epoch(height_before_update, time_of_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before);
            assert!(state.in_mem().update_epoch_blocks_delay.is_none());

            // Update should be enqueued for 2 blocks in the future starting at or after this height and time
            state.update_epoch(height_of_update, time_of_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before);
            assert_eq!(state.in_mem().update_epoch_blocks_delay, Some(2));

            // Increment the block height and time to simulate new blocks now
            let height_of_update = height_of_update + 1;
            let time_of_update = time_of_update + Duration::seconds(1);
            state.update_epoch(height_of_update, time_of_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before);
            assert_eq!(state.in_mem().update_epoch_blocks_delay, Some(1));

            let height_of_update = height_of_update + 1;
            let time_of_update = time_of_update + Duration::seconds(1);
            state.update_epoch(height_of_update, time_of_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before.next());
            assert!(state.in_mem().update_epoch_blocks_delay.is_none());
            // The next epoch's minimum duration should change
            assert_eq!(state.in_mem().next_epoch_min_start_height,
                height_of_update + parameters.epoch_duration.min_num_of_blocks);
            assert_eq!(state.in_mem().next_epoch_min_start_time,
                time_of_update + parameters.epoch_duration.min_duration);

            // Increment the block height and time once more to make sure things reset
            let height_of_update = height_of_update + 1;
            let time_of_update = time_of_update + Duration::seconds(1);
            state.update_epoch(height_of_update, time_of_update).unwrap();
            assert_eq!(state.in_mem().block.epoch, epoch_before.next());
        }
    }

    fn test_key_1() -> Key {
        Key::parse("testing1").unwrap()
    }

    fn test_key_2() -> Key {
        Key::parse("testing2").unwrap()
    }

    fn merkle_tree_key_filter(key: &Key) -> bool {
        key == &test_key_1()
    }

    #[test]
    fn test_writing_without_merklizing_or_diffs() {
        let mut state = TestState::default();
        assert_eq!(state.in_mem().block.height.0, 0);

        (state.0.merkle_tree_key_filter) = merkle_tree_key_filter;

        let key1 = test_key_1();
        let val1 = 1u64;
        let key2 = test_key_2();
        let val2 = 2u64;

        // Standard write of key-val-1
        state.write(&key1, val1).unwrap();

        // Read from State should return val1
        let res = state.read::<u64>(&key1).unwrap().unwrap();
        assert_eq!(res, val1);

        // Read from DB shouldn't return val1 bc the block hasn't been
        // committed
        let (res, _) = state.db_read(&key1).unwrap();
        assert!(res.is_none());

        // Write key-val-2 without merklizing or diffs
        state.write(&key2, val2).unwrap();

        // Read from state should return val2
        let res = state.read::<u64>(&key2).unwrap().unwrap();
        assert_eq!(res, val2);

        // Commit block and storage changes
        state.commit_block().unwrap();
        state.in_mem_mut().block.height =
            state.in_mem().block.height.next_height();

        // Read key1 from DB should return val1
        let (res1, _) = state.db_read(&key1).unwrap();
        let res1 = u64::try_from_slice(&res1.unwrap()).unwrap();
        assert_eq!(res1, val1);

        // Check merkle tree inclusion of key-val-1 explicitly
        let is_merklized1 = state.in_mem().block.tree.has_key(&key1).unwrap();
        assert!(is_merklized1);

        // Key2 should be in storage. Confirm by reading from
        // state and also by reading DB subspace directly
        let res2 = state.read::<u64>(&key2).unwrap().unwrap();
        assert_eq!(res2, val2);
        let res2 = state.db().read_subspace_val(&key2).unwrap().unwrap();
        let res2 = u64::try_from_slice(&res2).unwrap();
        assert_eq!(res2, val2);

        // Check explicitly that key-val-2 is not in merkle tree
        let is_merklized2 = state.in_mem().block.tree.has_key(&key2).unwrap();
        assert!(!is_merklized2);

        // Check that the proper diffs exist for key-val-1
        let res1 = state
            .db()
            .read_diffs_val(&key1, Default::default(), true)
            .unwrap();
        assert!(res1.is_none());

        let res1 = state
            .db()
            .read_diffs_val(&key1, Default::default(), false)
            .unwrap()
            .unwrap();
        let res1 = u64::try_from_slice(&res1).unwrap();
        assert_eq!(res1, val1);

        // Check that there are diffs for key-val-2 in block 0, since all keys
        // need to have diffs for at least 1 block for rollback purposes
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(0), true)
            .unwrap();
        assert!(res2.is_none());
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(0), false)
            .unwrap()
            .unwrap();
        let res2 = u64::try_from_slice(&res2).unwrap();
        assert_eq!(res2, val2);

        // Now delete the keys properly
        state.delete(&key1).unwrap();
        state.delete(&key2).unwrap();

        // Commit the block again
        state.commit_block().unwrap();
        state.in_mem_mut().block.height =
            state.in_mem().block.height.next_height();

        // Check the key-vals are removed from the storage subspace
        let res1 = state.read::<u64>(&key1).unwrap();
        let res2 = state.read::<u64>(&key2).unwrap();
        assert!(res1.is_none() && res2.is_none());
        let res1 = state.db().read_subspace_val(&key1).unwrap();
        let res2 = state.db().read_subspace_val(&key2).unwrap();
        assert!(res1.is_none() && res2.is_none());

        // Check that the key-vals don't exist in the merkle tree anymore
        let is_merklized1 = state.in_mem().block.tree.has_key(&key1).unwrap();
        let is_merklized2 = state.in_mem().block.tree.has_key(&key2).unwrap();
        assert!(!is_merklized1 && !is_merklized2);

        // Check that key-val-1 diffs are properly updated for blocks 0 and 1
        let res1 = state
            .db()
            .read_diffs_val(&key1, BlockHeight(0), true)
            .unwrap();
        assert!(res1.is_none());

        let res1 = state
            .db()
            .read_diffs_val(&key1, BlockHeight(0), false)
            .unwrap()
            .unwrap();
        let res1 = u64::try_from_slice(&res1).unwrap();
        assert_eq!(res1, val1);

        let res1 = state
            .db()
            .read_diffs_val(&key1, BlockHeight(1), true)
            .unwrap()
            .unwrap();
        let res1 = u64::try_from_slice(&res1).unwrap();
        assert_eq!(res1, val1);

        let res1 = state
            .db()
            .read_diffs_val(&key1, BlockHeight(1), false)
            .unwrap();
        assert!(res1.is_none());

        // Check that key-val-2 diffs don't exist for block 0 anymore
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(0), true)
            .unwrap();
        assert!(res2.is_none());
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(0), false)
            .unwrap();
        assert!(res2.is_none());

        // Check that the block 1 diffs for key-val-2 include an "old" value of
        // val2 and no "new" value
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(1), true)
            .unwrap()
            .unwrap();
        let res2 = u64::try_from_slice(&res2).unwrap();
        assert_eq!(res2, val2);
        let res2 = state
            .db()
            .read_diffs_val(&key2, BlockHeight(1), false)
            .unwrap();
        assert!(res2.is_none());
    }
}
