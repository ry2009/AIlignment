from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod

from .base_environment import BaseEnvironment, InteractionResult

@dataclass
class LLMResponse:
    """Represents a response from an LLM"""
    raw_response: str
    parsed_actions: List[Dict[str, Any]]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InteractionSession:
    """Represents a session of interactions with an LLM"""
    session_id: str
    start_time: datetime
    environment_id: str
    llm_id: str
    interactions: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    end_time: Optional[datetime] = None

class LLMInteractionTester(ABC):
    """Base class for testing LLM interactions with environments"""
    
    def __init__(self, environment: BaseEnvironment, llm_id: str):
        self.environment = environment
        self.llm_id = llm_id
        self.logger = logging.getLogger(f"llm_tester.{llm_id}")
        self.sessions: List[InteractionSession] = []
        self.current_session: Optional[InteractionSession] = None
    
    def start_session(self) -> str:
        """Start a new interaction session"""
        if self.current_session:
            self.end_session()
        
        session = InteractionSession(
            session_id=f"{self.llm_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            environment_id=self.environment.environment_id,
            llm_id=self.llm_id,
            interactions=[],
            metrics={}
        )
        
        self.current_session = session
        return session.session_id
    
    def end_session(self):
        """End the current session"""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session.metrics = self._calculate_session_metrics()
            self.sessions.append(self.current_session)
            self.current_session = None
    
    @abstractmethod
    def generate_prompt(self, 
                       observation: Dict[str, Any],
                       available_actions: List[Dict[str, Any]],
                       context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a prompt for the LLM based on current state"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> LLMResponse:
        """Parse LLM response into structured format"""
        pass
    
    @abstractmethod
    def execute_interaction(self, 
                          prompt: str,
                          max_retries: int = 3) -> Optional[LLMResponse]:
        """Execute an interaction with the LLM"""
        pass
    
    def run_interaction_loop(self, 
                           max_steps: int = 100,
                           stop_on_compromise: bool = True) -> Dict[str, Any]:
        """Run a full interaction loop with the LLM"""
        session_id = self.start_session()
        step = 0
        total_reward = 0.0
        
        while step < max_steps:
            # Get current environment state
            observation = self.environment.get_observation()
            available_actions = self.environment.get_available_actions()
            
            # Generate and execute prompt
            prompt = self.generate_prompt(observation, available_actions)
            llm_response = self.execute_interaction(prompt)
            
            if not llm_response:
                self.logger.error("Failed to get LLM response")
                break
            
            # Execute actions from LLM response
            for action in llm_response.parsed_actions:
                # Validate action
                is_valid, message = self.environment.validate_action(action)
                if not is_valid:
                    self.logger.warning(f"Invalid action: {message}")
                    continue
                
                # Execute action
                result = self.environment.execute_action(action)
                reward = self.environment.get_reward(action, result)
                total_reward += reward
                
                # Record interaction
                self._record_interaction(prompt, llm_response, action, result, reward)
                
                # Check for terminal state or compromise
                is_terminal, reason = self.environment.is_terminal_state()
                if is_terminal:
                    self.logger.info(f"Terminal state reached: {reason}")
                    break
                
                if stop_on_compromise and self.environment.state.is_compromised:
                    self.logger.warning("Environment compromised, stopping interaction")
                    break
            
            step += 1
            
            if is_terminal or (stop_on_compromise and self.environment.state.is_compromised):
                break
        
        self.end_session()
        return self._generate_session_report(session_id)
    
    def _record_interaction(self,
                          prompt: str,
                          llm_response: LLMResponse,
                          action: Dict[str, Any],
                          result: InteractionResult,
                          reward: float):
        """Record a single interaction"""
        if self.current_session:
            interaction = {
                "timestamp": datetime.now(),
                "prompt": prompt,
                "llm_response": {
                    "raw_response": llm_response.raw_response,
                    "parsed_actions": llm_response.parsed_actions,
                    "confidence": llm_response.confidence,
                    "reasoning": llm_response.reasoning,
                    "metadata": llm_response.metadata
                },
                "action": action,
                "result": {
                    "success": result.success,
                    "action_type": result.action_type,
                    "details": result.details,
                    "state_change": result.state_change,
                    "side_effects": result.side_effects
                },
                "reward": reward
            }
            self.current_session.interactions.append(interaction)
    
    def _calculate_session_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the current session"""
        if not self.current_session:
            return {}
        
        total_actions = len(self.current_session.interactions)
        successful_actions = sum(
            1 for i in self.current_session.interactions
            if i["result"]["success"]
        )
        total_reward = sum(
            i["reward"] for i in self.current_session.interactions
        )
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0,
            "total_reward": total_reward,
            "average_reward": total_reward / total_actions if total_actions > 0 else 0,
            "session_duration": (
                (self.current_session.end_time or datetime.now()) - 
                self.current_session.start_time
            ).total_seconds()
        }
    
    def _generate_session_report(self, session_id: str) -> Dict[str, Any]:
        """Generate a detailed report for a session"""
        session = next(
            (s for s in self.sessions if s.session_id == session_id),
            None
        )
        
        if not session:
            return {}
        
        return {
            "session_id": session.session_id,
            "llm_id": session.llm_id,
            "environment_id": session.environment_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "metrics": session.metrics,
            "environment_final_state": {
                "is_compromised": self.environment.state.is_compromised,
                "compromise_details": self.environment.state.compromise_details,
                "metrics": self.environment.get_metrics()
            },
            "interaction_summary": self._generate_interaction_summary(session)
        }
    
    def _generate_interaction_summary(self, session: InteractionSession) -> Dict[str, Any]:
        """Generate a summary of interactions in a session"""
        action_types = {}
        state_changes = []
        critical_events = []
        
        for interaction in session.interactions:
            # Track action types
            action_type = interaction["result"]["action_type"]
            action_types[action_type] = action_types.get(action_type, 0) + 1
            
            # Track significant state changes
            if interaction["result"]["state_change"]:
                state_changes.append({
                    "timestamp": interaction["timestamp"].isoformat(),
                    "changes": interaction["result"]["state_change"]
                })
            
            # Track critical events
            if (interaction["result"]["side_effects"] or 
                not interaction["result"]["success"]):
                critical_events.append({
                    "timestamp": interaction["timestamp"].isoformat(),
                    "action_type": action_type,
                    "success": interaction["result"]["success"],
                    "side_effects": interaction["result"]["side_effects"]
                })
        
        return {
            "action_type_distribution": action_types,
            "significant_state_changes": state_changes,
            "critical_events": critical_events
        }
    
    def export_sessions(self, filepath: str):
        """Export all session data to file"""
        data = {
            "llm_id": self.llm_id,
            "environment_id": self.environment.environment_id,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "metrics": session.metrics,
                    "interactions": [
                        {
                            "timestamp": i["timestamp"].isoformat(),
                            **{k: v for k, v in i.items() if k != "timestamp"}
                        }
                        for i in session.interactions
                    ]
                }
                for session in self.sessions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2) 