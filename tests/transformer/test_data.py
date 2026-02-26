# -*- coding: utf-8 -*-
"""
Data Loading Tests
==================

Tests for transformer.data module.
"""

import pytest
import torch


class TestCreateDataloaders:
    """Test create_dataloaders function."""

    @pytest.mark.slow
    def test_create_wikitext2_dataloaders(self):
        """Test creating WikiText-2 dataloaders."""
        from transformer.data import create_dataloaders

        train_loader, val_loader, vocab_size = create_dataloaders(
            max_seq_len=64,
            batch_size=4,
            dataset='wikitext-2',
        )

        assert train_loader is not None
        assert val_loader is not None
        assert vocab_size > 0

    @pytest.mark.slow
    def test_dataloader_batch_shapes(self):
        """Test dataloader produces correct batch shapes."""
        from transformer.data import create_dataloaders

        max_seq_len = 64
        batch_size = 4

        train_loader, val_loader, vocab_size = create_dataloaders(
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            dataset='wikitext-2',
        )

        # Get a batch
        batch = next(iter(train_loader))

        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            inputs = batch
            targets = None

        # Check shape
        assert inputs.shape[0] <= batch_size
        assert inputs.shape[1] == max_seq_len

    @pytest.mark.slow
    def test_dataloader_token_range(self):
        """Test tokens are in valid range."""
        from transformer.data import create_dataloaders

        train_loader, val_loader, vocab_size = create_dataloaders(
            max_seq_len=64,
            batch_size=4,
            dataset='wikitext-2',
        )

        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch

        assert (inputs >= 0).all()
        assert (inputs < vocab_size).all()


class TestCreateCharDataloaders:
    """Test create_char_dataloaders function."""

    @pytest.mark.slow
    def test_create_char_dataloaders(self):
        """Test creating character-level dataloaders."""
        from transformer.data import create_char_dataloaders

        train_loader, val_loader, vocab_size = create_char_dataloaders(
            max_seq_len=64,
            batch_size=4,
        )

        assert train_loader is not None
        assert val_loader is not None
        # Character vocab should be small (< 300)
        assert vocab_size < 300


class TestCreateByteDataloaders:
    """Test create_byte_dataloaders function."""

    @pytest.mark.slow
    def test_create_byte_dataloaders(self):
        """Test creating byte-level dataloaders."""
        from transformer.data import create_byte_dataloaders

        train_loader, val_loader, vocab_size = create_byte_dataloaders(
            max_seq_len=64,
            batch_size=4,
        )

        assert train_loader is not None
        assert val_loader is not None
        # Byte vocab is exactly 256
        assert vocab_size == 256


class TestWikiTextDataset:
    """Test WikiTextDataset backward-compatible alias."""

    @pytest.mark.slow
    def test_dataset_creation(self):
        """Test creating dataset via alias."""
        from transformer.data.datasets import WikiTextDataset

        dataset = WikiTextDataset(
            split='train',
            max_seq_len=64,
            dataset='wikitext-2',
        )

        assert len(dataset) > 0

    @pytest.mark.slow
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        from transformer.data.datasets import WikiTextDataset

        dataset = WikiTextDataset(
            split='train',
            max_seq_len=64,
            dataset='wikitext-2',
        )

        item = dataset[0]

        if isinstance(item, (list, tuple)):
            inputs, targets = item
            assert inputs.shape == (64,)
            assert targets.shape == (64,)
        else:
            assert item.shape == (64,)

    @pytest.mark.slow
    def test_dataset_has_tokenizer(self):
        """Test dataset has encode/decode methods."""
        from transformer.data.datasets import WikiTextDataset

        dataset = WikiTextDataset(
            split='train',
            max_seq_len=64,
            dataset='wikitext-2',
        )

        # Should have encode/decode for tokenization
        assert hasattr(dataset, 'encode') or hasattr(dataset, 'tokenizer')
