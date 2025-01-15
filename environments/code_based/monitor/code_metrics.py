import ast
import time
import cProfile
import io
import pstats
from typing import Dict, List, Any
from radon.metrics import mi_visit
from radon.complexity import cc_visit
from radon.raw import analyze

class CodeMetrics:
    def __init__(self):
        self.metrics_cache = {}
    
    def analyze_code(self, filepath: str) -> Dict[str, Any]:
        """Analyze code for various metrics"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        return {
            "quality_metrics": self._get_quality_metrics(content),
            "performance_metrics": self._get_performance_metrics(filepath),
            "complexity_metrics": self._get_complexity_metrics(content)
        }
    
    def _get_quality_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        # Calculate maintainability index
        maintainability_index = mi_visit(content, multi=True)
        
        # Get raw metrics
        raw_metrics = analyze(content)
        
        return {
            "maintainability_index": maintainability_index,
            "loc": raw_metrics.loc,
            "lloc": raw_metrics.lloc,
            "sloc": raw_metrics.sloc,
            "comments": raw_metrics.comments,
            "multi": raw_metrics.multi,
            "blank": raw_metrics.blank,
            "single_comments": raw_metrics.single_comments
        }
    
    def _get_complexity_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate code complexity metrics"""
        # Get cyclomatic complexity
        complexity_blocks = list(cc_visit(content))
        
        total_complexity = sum(block.complexity for block in complexity_blocks)
        avg_complexity = total_complexity / len(complexity_blocks) if complexity_blocks else 0
        
        return {
            "total_complexity": total_complexity,
            "average_complexity": avg_complexity,
            "complexity_blocks": [
                {
                    "name": block.name,
                    "complexity": block.complexity,
                    "lineno": block.lineno,
                    "type": block.type
                }
                for block in complexity_blocks
            ]
        }
    
    def _get_performance_metrics(self, filepath: str) -> Dict[str, Any]:
        """Measure code performance metrics"""
        # Profile code execution
        profiler = cProfile.Profile()
        
        # Import the module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        
        # Profile module loading
        profiler.enable()
        spec.loader.exec_module(module)
        profiler.disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Parse profiling results
        stats = ps.stats
        total_calls = sum(stats[func][0] for func in stats)
        total_time = sum(stats[func][3] for func in stats)
        
        return {
            "total_calls": total_calls,
            "total_time": total_time,
            "function_stats": [
                {
                    "function": f"{func[0]}:{func[1]}({func[2]})",
                    "calls": stats[func][0],
                    "time": stats[func][3]
                }
                for func in stats
            ][:10]  # Top 10 functions by time
        }
    
    def compare_metrics(self, original_metrics: Dict[str, Any], 
                       refactored_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics between original and refactored code"""
        return {
            "quality_diff": {
                key: refactored_metrics["quality_metrics"][key] - 
                     original_metrics["quality_metrics"][key]
                for key in original_metrics["quality_metrics"]
                if isinstance(original_metrics["quality_metrics"][key], (int, float))
            },
            "complexity_diff": {
                "total": refactored_metrics["complexity_metrics"]["total_complexity"] - 
                        original_metrics["complexity_metrics"]["total_complexity"],
                "average": refactored_metrics["complexity_metrics"]["average_complexity"] - 
                          original_metrics["complexity_metrics"]["average_complexity"]
            },
            "performance_diff": {
                "total_time": refactored_metrics["performance_metrics"]["total_time"] - 
                             original_metrics["performance_metrics"]["total_time"],
                "total_calls": refactored_metrics["performance_metrics"]["total_calls"] - 
                              original_metrics["performance_metrics"]["total_calls"]
            }
        } 