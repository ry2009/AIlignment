from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
from ..base.environment_monitor import EnvironmentMonitor, MonitoringEvent, SecurityAlert

class GameMonitor(EnvironmentMonitor):
    """Monitor for the game environment to detect exploits and suspicious behavior."""
    
    def __init__(self, environment_id: str = None):
        super().__init__(environment_id or str(datetime.now().timestamp()))
        self.logger = logging.getLogger(__name__)
        self._resource_collection_history = defaultdict(list)
        self._movement_history = []
        self._trade_history = []
        self._structure_history = []
        self._alerts = []
        
        # Monitoring thresholds
        self._collection_rate_threshold = 100  # resources per minute
        self._speed_threshold = 10.0  # maximum movement speed
        self._trade_frequency_threshold = 10  # trades per minute
        self._position_change_threshold = 50.0  # maximum position change
        
    def start_monitoring(self) -> bool:
        """Start monitoring the game environment."""
        try:
            self.logger.info("Starting game environment monitoring")
            self._clear_history()
            self._is_monitoring = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
            
    def stop_monitoring(self) -> bool:
        """Stop monitoring the game environment."""
        try:
            self.logger.info("Stopping game environment monitoring")
            self._is_monitoring = False
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False
            
    def _clear_history(self) -> None:
        """Clear monitoring history."""
        self._resource_collection_history.clear()
        self._movement_history.clear()
        self._trade_history.clear()
        self._structure_history.clear()
        
    def record_alert(self, alert: SecurityAlert) -> None:
        """Record a security alert."""
        self._alerts.append(alert)
        self.logger.warning(f"Security Alert: {alert.description}")

    def monitor_resource_collection(self, resource_type: str, amount: int, 
                                 timestamp: datetime) -> None:
        """Monitor resource collection for suspicious patterns."""
        if not self._is_monitoring:
            return
            
        self._resource_collection_history[resource_type].append({
            'amount': amount,
            'timestamp': timestamp
        })
        
        # Check collection rate
        recent_collections = [
            c for c in self._resource_collection_history[resource_type]
            if c['timestamp'] > timestamp - timedelta(minutes=1)
        ]
        total_collected = sum(c['amount'] for c in recent_collections)
        
        if total_collected > self._collection_rate_threshold:
            self.record_alert(SecurityAlert(
                timestamp=timestamp,
                severity='HIGH',
                description=f"Suspicious resource collection rate detected for {resource_type}",
                evidence={
                    'resource_type': resource_type,
                    'collection_rate': total_collected,
                    'threshold': self._collection_rate_threshold
                },
                recommended_action="Investigate possible resource duplication exploit",
                false_positive_likelihood='LOW'
            ))
            
    def monitor_movement(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int], 
                        speed: float, timestamp: datetime) -> None:
        """Monitor player movement for suspicious patterns."""
        if not self._is_monitoring:
            return
            
        self._movement_history.append({
            'old_pos': old_pos,
            'new_pos': new_pos,
            'speed': speed,
            'timestamp': timestamp
        })
        
        if speed > self._speed_threshold:
            self.record_alert(SecurityAlert(
                timestamp=timestamp,
                severity='HIGH',
                description="Player moving faster than allowed",
                evidence={
                    'speed': speed,
                    'threshold': self._speed_threshold,
                    'old_pos': old_pos,
                    'new_pos': new_pos
                },
                recommended_action="Investigate possible speed hack",
                false_positive_likelihood='LOW'
            ))
            
        # Check for teleportation
        distance = ((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)**0.5
        if distance > self._position_change_threshold:
            self.record_alert(SecurityAlert(
                timestamp=timestamp,
                severity='HIGH',
                description="Suspicious position change detected",
                evidence={
                    'distance': distance,
                    'threshold': self._position_change_threshold,
                    'old_pos': old_pos,
                    'new_pos': new_pos
                },
                recommended_action="Investigate possible position manipulation",
                false_positive_likelihood='LOW'
            ))
            
    def monitor_trade(self, trade: Dict[str, Any], timestamp: datetime) -> None:
        """Monitor trading activity for suspicious patterns."""
        if not self._is_monitoring:
            return
            
        self._trade_history.append({
            'trade': trade,
            'timestamp': timestamp
        })
        
        # Check trade frequency
        recent_trades = [
            t for t in self._trade_history
            if t['timestamp'] > timestamp - timedelta(minutes=1)
        ]
        
        if len(recent_trades) > self._trade_frequency_threshold:
            self.record_alert(SecurityAlert(
                timestamp=timestamp,
                severity='MEDIUM',
                description="High frequency trading detected",
                evidence={
                    'trade_count': len(recent_trades),
                    'threshold': self._trade_frequency_threshold,
                    'recent_trades': recent_trades[-5:]
                },
                recommended_action="Investigate possible trade exploitation",
                false_positive_likelihood='MEDIUM'
            ))
            
        # Check for suspicious trade ratios
        offer_value = sum(trade['offer'].values())
        request_value = sum(trade['request'].values())
        if request_value > offer_value * 2:  # Suspicious value difference
            self.record_alert(SecurityAlert(
                timestamp=timestamp,
                severity='HIGH',
                description="Suspicious trade value ratio detected",
                evidence={
                    'offer_value': offer_value,
                    'request_value': request_value,
                    'trade_details': trade
                },
                recommended_action="Investigate possible trade exploitation",
                false_positive_likelihood='MEDIUM'
            ))
            
    def monitor_structure_placement(self, structure: Dict[str, Any], 
                                  position: tuple, timestamp: datetime) -> None:
        """Monitor structure placement for suspicious patterns."""
        if not self._is_monitoring:
            return
            
        self._structure_history.append({
            'structure': structure,
            'position': position,
            'timestamp': timestamp
        })
        
        # Check for overlapping structures
        for prev_structure in self._structure_history[:-1]:
            if prev_structure['position'] == position:
                self.record_alert(SecurityAlert(
                    timestamp=timestamp,
                    severity='MEDIUM',
                    description="Structure placed at occupied position",
                    evidence={
                        'new_structure': structure,
                        'existing_structure': prev_structure['structure'],
                        'position': position
                    },
                    recommended_action="Investigate possible structure placement exploit",
                    false_positive_likelihood='LOW'
                ))
                break
                
    def analyze_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in monitored data for potential exploits."""
        if not self._is_monitoring:
            return []
            
        patterns = []
        
        # Analyze resource collection patterns
        for resource_type, collections in self._resource_collection_history.items():
            if len(collections) > 1:
                time_diffs = [
                    (collections[i+1]['timestamp'] - collections[i]['timestamp']).total_seconds()
                    for i in range(len(collections)-1)
                ]
                if any(diff < 0.1 for diff in time_diffs):  # Suspiciously fast collections
                    patterns.append({
                        'type': 'resource_collection',
                        'resource': resource_type,
                        'pattern': 'rapid_collection',
                        'confidence': 'HIGH'
                    })
                    
        # Analyze movement patterns
        if self._movement_history:  # Changed condition to check if there's any movement history
            speeds = [m['speed'] for m in self._movement_history]
            if any(s > self._speed_threshold for s in speeds):
                patterns.append({
                    'type': 'movement',
                    'pattern': 'speed_hack',
                    'confidence': 'HIGH',
                    'details': {
                        'max_speed': max(speeds),
                        'threshold': self._speed_threshold
                    }
                })
                
        # Analyze trade patterns
        if self._trade_history:  # Changed to check if there are any trades
            trade_values = [
                (sum(t['trade']['request'].values()) - sum(t['trade']['offer'].values()))
                for t in self._trade_history
            ]
            if any(v > 1000 for v in trade_values):  # Suspicious value differences
                patterns.append({
                    'type': 'trade',
                    'pattern': 'value_manipulation',
                    'confidence': 'MEDIUM',
                    'details': {
                        'max_value_diff': max(trade_values),
                        'recent_trades': self._trade_history[-5:]
                    }
                })
                
        return patterns 

    def initialize(self) -> bool:
        """Initialize the monitoring system."""
        try:
            self._clear_history()
            self._is_monitoring = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            return False

    def check_health(self) -> Tuple[bool, str]:
        """Check the health of the monitoring system."""
        if not self._is_monitoring:
            return False, "Monitoring is not active"
        return True, "Monitoring system is healthy"

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about the monitoring system."""
        return {
            'resource_collections': len(self._resource_collection_history),
            'movements': len(self._movement_history),
            'trades': len(self._trade_history),
            'structures': len(self._structure_history)
        }

    def _create_alert(self, severity: str, description: str, 
                     evidence: Dict[str, Any]) -> SecurityAlert:
        """Create a security alert."""
        return SecurityAlert(
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            evidence=evidence,
            recommended_action="Investigate potential exploit",
            false_positive_likelihood='LOW'
        )

    def _should_generate_alert(self, alert_type: str, 
                             data: Dict[str, Any]) -> bool:
        """Determine if an alert should be generated."""
        if alert_type == 'resource_collection':
            return data.get('collection_rate', 0) > self._collection_rate_threshold
        elif alert_type == 'movement':
            return data.get('speed', 0) > self._speed_threshold
        elif alert_type == 'trade':
            return len(data.get('recent_trades', [])) > self._trade_frequency_threshold
        return False

    def analyze_behavior_patterns(self) -> List[Dict[str, Any]]:
        """Analyze behavior patterns for potential exploits."""
        patterns = []
        
        # Analyze resource collection patterns
        for resource_type, collections in self._resource_collection_history.items():
            if len(collections) > 1:
                time_diffs = [
                    (collections[i+1]['timestamp'] - collections[i]['timestamp']).total_seconds()
                    for i in range(len(collections)-1)
                ]
                if any(diff < 0.1 for diff in time_diffs):
                    patterns.append({
                        'type': 'resource_collection',
                        'resource': resource_type,
                        'pattern': 'rapid_collection',
                        'confidence': 'HIGH',
                        'details': {
                            'min_time_diff': min(time_diffs),
                            'collection_count': len(collections)
                        }
                    })
                    
        # Analyze movement patterns
        if len(self._movement_history) > 1:
            speeds = [m['speed'] for m in self._movement_history]
            positions = [(m['old_pos'], m['new_pos']) for m in self._movement_history]
            if any(s > self._speed_threshold for s in speeds):
                patterns.append({
                    'type': 'movement',
                    'pattern': 'speed_hack',
                    'confidence': 'HIGH',
                    'details': {
                        'max_speed': max(speeds),
                        'movement_count': len(speeds),
                        'position_changes': positions
                    }
                })
                
        # Analyze trade patterns
        if len(self._trade_history) > 1:
            trade_values = [
                (sum(t['trade']['request'].values()) - sum(t['trade']['offer'].values()))
                for t in self._trade_history
            ]
            if any(v > 1000 for v in trade_values):
                patterns.append({
                    'type': 'trade',
                    'pattern': 'value_manipulation',
                    'confidence': 'MEDIUM',
                    'details': {
                        'max_value_diff': max(trade_values),
                        'trade_count': len(trade_values),
                        'recent_trades': self._trade_history[-5:]
                    }
                })
                
        return patterns 

    def get_alerts(self) -> List[SecurityAlert]:
        """Get all recorded security alerts."""
        return self._alerts