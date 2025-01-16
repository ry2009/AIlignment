from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from datetime import datetime

def check_usage_pattern(
    usage: np.ndarray,
    rules: Dict[str, Any]
) -> bool:
    """Check if resource usage follows expected patterns."""
    # Check for sudden spikes
    if 'max_spike' in rules:
        diffs = np.diff(usage, axis=1)
        if np.max(np.abs(diffs)) > rules['max_spike']:
            return False
    
    # Check for sustained high usage
    if 'max_sustained' in rules:
        window = rules.get('sustained_window', 5)
        rolling_mean = np.mean([usage[:, i:i+window] for i in range(usage.shape[1]-window+1)], axis=2)
        if np.any(rolling_mean > rules['max_sustained']):
            return False
    
    # Check for periodic patterns
    if 'periodic_check' in rules and rules['periodic_check']:
        # Use FFT to detect periodic patterns
        fft = np.fft.fft(usage, axis=1)
        freqs = np.fft.fftfreq(usage.shape[1])
        main_freq = freqs[np.argmax(np.abs(fft), axis=1)]
        if np.any(main_freq > rules.get('max_frequency', 0.5)):
            return False
    
    return True

def check_content_distribution(
    posts: List[Dict[str, Any]],
    rules: Dict[str, Any]
) -> bool:
    """Check if forum content follows expected distribution."""
    # Check topic distribution
    if 'topic_distribution' in rules:
        topic_counts = {}
        for post in posts:
            topic_counts[post['topic']] = topic_counts.get(post['topic'], 0) + 1
        
        for topic, expected_ratio in rules['topic_distribution'].items():
            actual_ratio = topic_counts.get(topic, 0) / len(posts)
            if abs(actual_ratio - expected_ratio) > rules.get('topic_tolerance', 0.1):
                return False
    
    # Check content length distribution
    if 'length_distribution' in rules:
        lengths = [post['content_length'] for post in posts]
        if rules['length_distribution'] == 'normal':
            _, p_value = stats.normaltest(lengths)
            if p_value < rules.get('p_threshold', 0.05):
                return False
    
    # Check engagement distribution
    if 'engagement_distribution' in rules:
        scores = [post['engagement_score'] for post in posts]
        if rules['engagement_distribution'] == 'uniform':
            _, p_value = stats.kstest(scores, 'uniform')
            if p_value < rules.get('p_threshold', 0.05):
                return False
    
    return True

def check_game_balance(
    episodes: List[Dict[str, Any]],
    rules: Dict[str, Any]
) -> bool:
    """Check if game mechanics are balanced."""
    # Check win rate by difficulty
    if 'win_rate_by_difficulty' in rules:
        for difficulty, expected_rate in rules['win_rate_by_difficulty'].items():
            diff_episodes = [e for e in episodes if e['difficulty'] == difficulty]
            if diff_episodes:
                actual_rate = sum(1 for e in diff_episodes if e['won']) / len(diff_episodes)
                if abs(actual_rate - expected_rate) > rules.get('win_rate_tolerance', 0.1):
                    return False
    
    # Check reward distribution
    if 'reward_distribution' in rules:
        rewards = [e['rewards'].mean() for e in episodes]
        if rules['reward_distribution'] == 'normal':
            _, p_value = stats.normaltest(rewards)
            if p_value < rules.get('p_threshold', 0.05):
                return False
    
    # Check completion time balance
    if 'completion_time_range' in rules:
        times = [e['completion_time'] for e in episodes]
        min_time, max_time = rules['completion_time_range']
        if min(times) < min_time or max(times) > max_time:
            return False
    
    return True

def check_feature_distribution(
    features: np.ndarray,
    rules: Dict[str, Any]
) -> bool:
    """Check if features follow expected distribution."""
    if 'distribution_type' not in rules:
        return True
    
    if rules['distribution_type'] == 'normal':
        # Check each feature
        for feature_idx in range(features.shape[1]):
            _, p_value = stats.normaltest(features[:, feature_idx])
            if p_value < rules.get('p_threshold', 0.05):
                return False
    
    elif rules['distribution_type'] == 'uniform':
        # Check each feature
        for feature_idx in range(features.shape[1]):
            _, p_value = stats.kstest(features[:, feature_idx], 'uniform')
            if p_value < rules.get('p_threshold', 0.05):
                return False
    
    return True

def check_label_distribution(
    labels: np.ndarray,
    rules: Dict[str, Any]
) -> bool:
    """Check if labels follow expected distribution."""
    if 'class_balance' in rules:
        # For classification tasks
        unique, counts = np.unique(labels, return_counts=True)
        class_ratios = counts / len(labels)
        
        for class_idx, expected_ratio in enumerate(rules['class_balance']):
            if abs(class_ratios[class_idx] - expected_ratio) > rules.get('balance_tolerance', 0.1):
                return False
    
    if 'value_range' in rules:
        # For regression tasks
        min_val, max_val = rules['value_range']
        if np.min(labels) < min_val or np.max(labels) > max_val:
            return False
    
    return True

def check_feature_correlation(
    features: np.ndarray,
    rules: Dict[str, Any]
) -> bool:
    """Check feature correlations."""
    if 'max_correlation' in rules:
        corr_matrix = np.corrcoef(features.T)
        # Remove diagonal elements
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        max_corr = np.max(np.abs(corr_matrix[mask]))
        
        if max_corr > rules['max_correlation']:
            return False
    
    return True

def validate_admin_forum_interaction(
    admin_data: Dict[str, Any],
    forum_data: Dict[str, Any],
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate interactions between admin and forum environments."""
    results = {}
    
    # Check if high resource usage correlates with high forum activity
    if 'resource_activity_correlation' in rules:
        resource_usage = np.mean([r['usage'] for r in admin_data['resources'].values()], axis=0)
        post_times = [datetime.fromtimestamp(p['timestamp']) for p in forum_data['posts']]
        post_counts = np.zeros(len(admin_data['timestamps']))
        
        for post_time in post_times:
            time_idx = np.argmin(np.abs(np.array(admin_data['timestamps']) - post_time.timestamp()))
            post_counts[time_idx] += 1
        
        correlation = np.corrcoef(resource_usage, post_counts)[0, 1]
        results['resource_activity_correlation'] = float(correlation)
        results['correlation_valid'] = abs(correlation) < rules['resource_activity_correlation']
    
    return results

def validate_forum_game_interaction(
    forum_data: Dict[str, Any],
    game_data: Dict[str, Any],
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate interactions between forum and game environments."""
    results = {}
    
    # Check if game performance correlates with forum activity
    if 'performance_activity_correlation' in rules:
        game_performance = np.array([e['rewards'].mean() for e in game_data['episodes']])
        player_posts = {p['player_id']: [] for p in game_data['players']}
        
        for post in forum_data['posts']:
            if post['user_id'] in player_posts:
                player_posts[post['user_id']].append(post)
        
        post_counts = np.array([len(posts) for posts in player_posts.values()])
        correlation = np.corrcoef(game_performance, post_counts)[0, 1]
        
        results['performance_activity_correlation'] = float(correlation)
        results['correlation_valid'] = abs(correlation) < rules['performance_activity_correlation']
    
    return results

def validate_game_ml_interaction(
    game_data: Dict[str, Any],
    ml_data: Dict[str, Any],
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate interactions between game and ML environments."""
    results = {}
    
    # Check if ML predictions align with game outcomes
    if 'prediction_accuracy' in rules:
        game_outcomes = np.array([e['won'] for e in game_data['episodes']])
        ml_predictions = ml_data.get('predictions', np.array([]))
        
        if len(game_outcomes) == len(ml_predictions):
            accuracy = np.mean(game_outcomes == ml_predictions)
            results['prediction_accuracy'] = float(accuracy)
            results['accuracy_valid'] = accuracy >= rules['prediction_accuracy']
    
    return results 