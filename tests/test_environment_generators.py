import pytest
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime
from ml_environment.testing.data.environment_generators import (
    AdminDataGenerator,
    ForumDataGenerator,
    GameDataGenerator,
    AdminDataConfig,
    ForumDataConfig,
    GameDataConfig
)

@pytest.fixture
def temp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def admin_generator(temp_path):
    return AdminDataGenerator(temp_path)

@pytest.fixture
def forum_generator(temp_path):
    return ForumDataGenerator(temp_path)

@pytest.fixture
def game_generator(temp_path):
    return GameDataGenerator(temp_path)

# Admin Environment Tests
def test_admin_data_generation(admin_generator):
    """Test generating administrative validation data."""
    config = AdminDataConfig(
        size=100,
        resource_types=['cpu', 'memory', 'disk'],
        user_count=50,
        time_periods=24,
        anomaly_rate=0.05,
        seed=42
    )
    
    data, version = admin_generator.generate_data(config)
    
    # Check data structure
    assert 'resources' in data
    assert 'users' in data
    assert 'timestamps' in data
    
    # Check resource data
    for resource_type in config.resource_types:
        assert resource_type in data['resources']
        assert data['resources'][resource_type]['usage'].shape == (config.size, config.time_periods)
        assert data['resources'][resource_type]['alerts'].shape == (config.size, config.time_periods)
    
    # Check user data
    assert len(data['users']['user_ids']) == config.user_count
    assert data['users']['login_times'].shape == (config.user_count, config.time_periods)
    
    # Check version metadata
    assert version.environment == "admin"
    assert 'admin' in version.tags
    assert 'validation' in version.tags

def test_admin_anomaly_generation(admin_generator):
    """Test anomaly generation in administrative data."""
    config = AdminDataConfig(
        size=1000,
        resource_types=['cpu'],
        user_count=10,
        time_periods=24,
        anomaly_rate=0.1,
        seed=42
    )
    
    data, _ = admin_generator.generate_data(config)
    
    # Check anomaly rate
    alerts = data['resources']['cpu']['alerts']
    actual_anomaly_rate = alerts.sum() / alerts.size
    assert abs(actual_anomaly_rate - config.anomaly_rate) < 0.02  # Allow small deviation

# Forum Environment Tests
def test_forum_data_generation(forum_generator):
    """Test generating forum validation data."""
    config = ForumDataConfig(
        size=100,
        num_users=20,
        topics=['tech', 'science', 'gaming'],
        post_length_range=(50, 500),
        toxic_content_rate=0.1,
        seed=42
    )
    
    data, version = forum_generator.generate_data(config)
    
    # Check data structure
    assert 'users' in data
    assert 'posts' in data
    assert 'topics' in data
    
    # Check users
    assert len(data['users']) == config.num_users
    assert all('toxicity_tendency' in user for user in data['users'])
    
    # Check posts
    assert len(data['posts']) == config.size
    for post in data['posts']:
        assert post['topic'] in config.topics
        assert config.post_length_range[0] <= post['content_length'] <= config.post_length_range[1]
    
    # Check version metadata
    assert version.environment == "forum"
    assert 'forum' in version.tags

def test_forum_toxicity_distribution(forum_generator):
    """Test toxicity distribution in forum data."""
    config = ForumDataConfig(
        size=1000,
        num_users=50,
        topics=['general'],
        post_length_range=(10, 100),
        toxic_content_rate=0.15,
        seed=42
    )
    
    data, _ = forum_generator.generate_data(config)
    
    # Check toxic content rate
    toxic_posts = sum(1 for post in data['posts'] if post['is_toxic'])
    actual_toxic_rate = toxic_posts / len(data['posts'])
    assert abs(actual_toxic_rate - config.toxic_content_rate) < 0.05  # Allow small deviation

# Game Environment Tests
def test_game_data_generation(game_generator):
    """Test generating game validation data."""
    config = GameDataConfig(
        num_episodes=100,
        episode_length=50,
        num_players=20,
        difficulty_levels=['easy', 'medium', 'hard'],
        win_rate_range=(0.4, 0.6),
        seed=42
    )
    
    data, version = game_generator.generate_data(config)
    
    # Check data structure
    assert 'players' in data
    assert 'episodes' in data
    assert 'difficulty_levels' in data
    
    # Check players
    assert len(data['players']) == config.num_players
    assert all('skill_level' in player for player in data['players'])
    
    # Check episodes
    assert len(data['episodes']) == config.num_episodes
    for episode in data['episodes']:
        assert episode['difficulty'] in config.difficulty_levels
        assert episode['actions'].shape[1] == config.episode_length
        assert episode['rewards'].shape[1] == config.episode_length
        assert episode['states'].shape == (config.episode_length, 10)
    
    # Check version metadata
    assert version.environment == "game"
    assert 'game' in version.tags

def test_game_win_rate_distribution(game_generator):
    """Test win rate distribution in game data."""
    config = GameDataConfig(
        num_episodes=500,
        episode_length=20,
        num_players=10,
        difficulty_levels=['medium'],
        win_rate_range=(0.45, 0.55),
        seed=42
    )
    
    data, _ = game_generator.generate_data(config)
    
    # Check win rate
    wins = sum(1 for episode in data['episodes'] if episode['won'])
    win_rate = wins / len(data['episodes'])
    assert config.win_rate_range[0] <= win_rate <= config.win_rate_range[1]

def test_data_reproducibility():
    """Test data reproducibility across all generators with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        
        # Test admin data
        admin_gen = AdminDataGenerator(path)
        config1 = AdminDataConfig(size=10, resource_types=['cpu'], user_count=5, 
                                time_periods=10, seed=42)
        config2 = AdminDataConfig(size=10, resource_types=['cpu'], user_count=5, 
                                time_periods=10, seed=42)
        
        data1, _ = admin_gen.generate_data(config1)
        data2, _ = admin_gen.generate_data(config2)
        assert np.allclose(data1['resources']['cpu']['usage'], 
                         data2['resources']['cpu']['usage'])
        
        # Test forum data
        forum_gen = ForumDataGenerator(path)
        config1 = ForumDataConfig(size=10, num_users=5, topics=['test'], 
                                post_length_range=(10, 20), seed=42)
        config2 = ForumDataConfig(size=10, num_users=5, topics=['test'], 
                                post_length_range=(10, 20), seed=42)
        
        data1, _ = forum_gen.generate_data(config1)
        data2, _ = forum_gen.generate_data(config2)
        assert len(data1['posts']) == len(data2['posts'])
        assert all(p1['content_length'] == p2['content_length'] 
                  for p1, p2 in zip(data1['posts'], data2['posts']))
        
        # Test game data
        game_gen = GameDataGenerator(path)
        config1 = GameDataConfig(num_episodes=10, episode_length=5, num_players=4,
                               difficulty_levels=['test'], win_rate_range=(0.4, 0.6), seed=42)
        config2 = GameDataConfig(num_episodes=10, episode_length=5, num_players=4,
                               difficulty_levels=['test'], win_rate_range=(0.4, 0.6), seed=42)
        
        data1, _ = game_gen.generate_data(config1)
        data2, _ = game_gen.generate_data(config2)
        assert len(data1['episodes']) == len(data2['episodes'])
        assert all(np.allclose(e1['actions'], e2['actions']) 
                  for e1, e2 in zip(data1['episodes'], data2['episodes'])) 