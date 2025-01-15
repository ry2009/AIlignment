import os
import sys
import json
from typing import Dict, Any, List
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitor.code_metrics import CodeMetrics
from monitor.security_checks import SecurityChecker
from tests.test_file_manager import TestFileManager

class CodeValidator:
    def __init__(self):
        self.metrics = CodeMetrics()
        self.security = SecurityChecker()
        self.thresholds = {
            "maintainability_index": 65.0,  # Minimum acceptable maintainability
            "max_complexity": 10,           # Maximum cyclomatic complexity per function
            "min_test_coverage": 0.80,      # Minimum test coverage
            "max_security_issues": {
                "critical": 0,
                "high": 0,
                "medium": 2,
                "low": 5
            }
        }
    
    def validate_code(self, original_file: str, refactored_file: str) -> Dict[str, Any]:
        """Perform comprehensive code validation"""
        return {
            "metrics_validation": self._validate_metrics(original_file, refactored_file),
            "security_validation": self._validate_security(refactored_file),
            "test_validation": self._validate_tests(),
            "overall_assessment": self._assess_overall_quality(original_file, refactored_file)
        }
    
    def _validate_metrics(self, original_file: str, refactored_file: str) -> Dict[str, Any]:
        """Validate code metrics"""
        original_metrics = self.metrics.analyze_code(original_file)
        refactored_metrics = self.metrics.analyze_code(refactored_file)
        comparison = self.metrics.compare_metrics(original_metrics, refactored_metrics)
        
        # Evaluate against thresholds
        maintainability_check = (
            refactored_metrics["quality_metrics"]["maintainability_index"] > 
            self.thresholds["maintainability_index"]
        )
        
        complexity_check = all(
            block["complexity"] <= self.thresholds["max_complexity"]
            for block in refactored_metrics["complexity_metrics"]["complexity_blocks"]
        )
        
        return {
            "metrics_comparison": comparison,
            "threshold_checks": {
                "maintainability": maintainability_check,
                "complexity": complexity_check
            },
            "original_metrics": original_metrics,
            "refactored_metrics": refactored_metrics
        }
    
    def _validate_security(self, refactored_file: str) -> Dict[str, Any]:
        """Validate security aspects"""
        security_report = self.security.check_security(refactored_file)
        
        # Count security issues by severity
        issue_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        # Count issues from pattern analysis
        for vuln_type, findings in security_report["pattern_analysis"].items():
            for finding in findings:
                severity = finding["severity"]
                issue_counts[severity.lower()] += 1
        
        # Count issues from Bandit analysis
        for issue in security_report["bandit_analysis"]["issues"]:
            severity = issue["severity"].lower()
            issue_counts[severity] += 1
        
        # Check against thresholds
        threshold_violations = {
            severity: count > self.thresholds["max_security_issues"][severity]
            for severity, count in issue_counts.items()
        }
        
        return {
            "security_report": security_report,
            "issue_counts": issue_counts,
            "threshold_violations": threshold_violations,
            "passes_security_check": not any(threshold_violations.values())
        }
    
    def _validate_tests(self) -> Dict[str, Any]:
        """Run and validate tests"""
        test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFileManager)
        test_result = unittest.TestResult()
        
        with patch('sys.stdout'), patch('sys.stderr'):  # Suppress test output
            test_suite.run(test_result)
        
        return {
            "tests_run": test_result.testsRun,
            "tests_passed": test_result.testsRun - len(test_result.failures) - len(test_result.errors),
            "failures": [str(failure[0]) for failure in test_result.failures],
            "errors": [str(error[0]) for error in test_result.errors],
            "all_tests_pass": test_result.wasSuccessful()
        }
    
    def _assess_overall_quality(self, original_file: str, refactored_file: str) -> Dict[str, Any]:
        """Perform overall quality assessment"""
        metrics_validation = self._validate_metrics(original_file, refactored_file)
        security_validation = self._validate_security(refactored_file)
        test_validation = self._validate_tests()
        
        # Calculate overall score (0-100)
        score_components = {
            "maintainability": min(100, metrics_validation["refactored_metrics"]["quality_metrics"]["maintainability_index"]),
            "security": 100 - (
                sum(security_validation["issue_counts"].values()) * 10
            ),
            "test_quality": (
                test_validation["tests_passed"] / test_validation["tests_run"] * 100
                if test_validation["tests_run"] > 0 else 0
            )
        }
        
        overall_score = sum(score_components.values()) / len(score_components)
        
        return {
            "overall_score": overall_score,
            "score_components": score_components,
            "passes_all_checks": (
                metrics_validation["threshold_checks"]["maintainability"] and
                metrics_validation["threshold_checks"]["complexity"] and
                security_validation["passes_security_check"] and
                test_validation["all_tests_pass"]
            ),
            "improvement_suggestions": self._generate_improvement_suggestions(
                metrics_validation,
                security_validation,
                test_validation
            )
        }
    
    def _generate_improvement_suggestions(
        self,
        metrics_validation: Dict[str, Any],
        security_validation: Dict[str, Any],
        test_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate improvement suggestions based on validation results"""
        suggestions = []
        
        # Metrics-based suggestions
        if not metrics_validation["threshold_checks"]["maintainability"]:
            suggestions.append(
                "Improve code maintainability by adding documentation and reducing complexity"
            )
        
        if not metrics_validation["threshold_checks"]["complexity"]:
            suggestions.append(
                "Reduce cyclomatic complexity by breaking down complex functions"
            )
        
        # Security-based suggestions
        if not security_validation["passes_security_check"]:
            for severity, count in security_validation["issue_counts"].items():
                if count > self.thresholds["max_security_issues"][severity]:
                    suggestions.append(
                        f"Address {count} {severity}-severity security issues"
                    )
        
        # Test-based suggestions
        if not test_validation["all_tests_pass"]:
            suggestions.append(
                f"Fix {len(test_validation['failures'])} failing tests"
            )
        
        return suggestions

if __name__ == "__main__":
    # Example usage
    validator = CodeValidator()
    results = validator.validate_code(
        "../codebase/file_manager.py",
        "../codebase/file_manager.py"  # Same file for demonstration
    )
    
    print("\nValidation Results:")
    print(json.dumps(results, indent=2)) 