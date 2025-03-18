import pytest
from unittest.mock import MagicMock
from src.config import Config
from src.networks.network import Network
from src.utils.shared_storage import SharedStorage

@pytest.fixture
def mock_network():
    """Fixture to create a mock Network instance."""
    return MagicMock(spec=Network)

def test_latest_network_returns_new_network_if_empty():
    """Test that latest_network() returns a new Network if storage is empty."""
    storage = SharedStorage()
    network = storage.latest_network()
    
    assert isinstance(network, Network), "latest_network() should return a Network instance"
    assert network.config == Config(), "The returned network should be initialized with a default Config"

def test_save_network_and_retrieve_latest(mock_network):
    """Test saving a network and retrieving the latest stored network."""
    storage = SharedStorage()

    # Mock networks for step 1 and step 2
    mock_network1 = MagicMock(spec=Network)
    mock_network2 = MagicMock(spec=Network)

    # Save networks at different steps
    storage.save_network(1, mock_network1)
    storage.save_network(2, mock_network2)

    # The latest network should be the one saved at step 2
    latest_network = storage.latest_network()
    
    assert latest_network == mock_network2, "latest_network() should return the most recently saved network"

def test_saving_multiple_networks_and_retrieving():
    """Test saving multiple networks and ensuring retrieval is correct."""
    storage = SharedStorage()

    mock_network1 = MagicMock(spec=Network)
    mock_network2 = MagicMock(spec=Network)
    mock_network3 = MagicMock(spec=Network)

    storage.save_network(10, mock_network1)
    storage.save_network(20, mock_network2)
    storage.save_network(15, mock_network3)

    latest_network = storage.latest_network()

    assert latest_network == mock_network2, "The latest network should be the one with the highest step number (20)"

def test_latest_network_does_not_modify_storage():
    """Test that calling latest_network() does not modify storage."""
    storage = SharedStorage()
    storage.latest_network()
    
    assert len(storage._networks) == 0, "Calling latest_network() should not modify the storage"
