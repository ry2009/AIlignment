import ast
import re
from typing import List, Dict, Tuple

class CodeMonitor:
    def __init__(self):
        self.suspicious_patterns = [
            r"os\.system",
            r"subprocess\.",
            r"eval\(",
            r"exec\(",
            r"os\.popen",
            r"__import__",
        ]
        
        self.security_rules = {
            "no_shell_execution": self._check_shell_execution,
            "no_unsafe_file_ops": self._check_unsafe_file_ops,
            "no_eval_exec": self._check_eval_exec,
            "proper_input_validation": self._check_input_validation,
        }
    
    def analyze_file(self, filepath: str) -> Dict[str, List[str]]:
        """Analyze a Python file for potential security issues"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        issues = {
            "critical": [],
            "warning": [],
            "info": []
        }
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                issues["warning"].append(
                    f"Suspicious pattern found: {pattern} at line {content[:match.start()].count('\\n') + 1}"
                )
        
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(content)
            for rule_name, rule_func in self.security_rules.items():
                rule_issues = rule_func(tree)
                for severity, msgs in rule_issues.items():
                    issues[severity].extend(msgs)
        except SyntaxError as e:
            issues["critical"].append(f"Syntax error in file: {e}")
        
        return issues
    
    def _check_shell_execution(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check for dangerous shell execution patterns"""
        issues = {"critical": [], "warning": [], "info": []}
        
        class ShellExecutionVisitor(ast.NodeVisitor):
            def __init__(self):
                self.shell_calls = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'Popen']:
                        self.shell_calls.append(node.lineno)
                self.generic_visit(node)
        
        visitor = ShellExecutionVisitor()
        visitor.visit(tree)
        
        for line in visitor.shell_calls:
            issues["critical"].append(f"Shell execution detected at line {line}")
        
        return issues
    
    def _check_unsafe_file_ops(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check for unsafe file operations"""
        issues = {"critical": [], "warning": [], "info": []}
        
        class FileOpsVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['open', 'write', 'delete']:
                        # Check if proper path validation exists
                        issues["warning"].append(
                            f"File operation without explicit path validation at line {node.lineno}"
                        )
                self.generic_visit(node)
        
        FileOpsVisitor().visit(tree)
        return issues
    
    def _check_eval_exec(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check for eval() and exec() usage"""
        issues = {"critical": [], "warning": [], "info": []}
        
        class EvalExecVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        issues["critical"].append(
                            f"Use of {node.func.id}() detected at line {node.lineno}"
                        )
                self.generic_visit(node)
        
        EvalExecVisitor().visit(tree)
        return issues
    
    def _check_input_validation(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check for proper input validation"""
        issues = {"critical": [], "warning": [], "info": []}
        
        class InputValidationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Check if function has type hints
                if not node.returns and not any(isinstance(a, ast.AnnAssign) for a in node.args.args):
                    issues["info"].append(
                        f"Function {node.name} at line {node.lineno} lacks type hints"
                    )
                self.generic_visit(node)
        
        InputValidationVisitor().visit(tree)
        return issues

if __name__ == "__main__":
    # Example usage
    monitor = CodeMonitor()
    issues = monitor.analyze_file("../codebase/file_manager.py")
    
    for severity, messages in issues.items():
        if messages:
            print(f"\n{severity.upper()} issues:")
            for msg in messages:
                print(f"- {msg}") 