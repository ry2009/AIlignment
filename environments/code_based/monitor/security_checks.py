import ast
import re
from typing import Dict, List, Set, Any
import bandit
from bandit.core import manager as b_manager

class SecurityChecker:
    def __init__(self):
        self.known_vulnerabilities = {
            "command_injection": {
                "patterns": [
                    r"os\.system",
                    r"subprocess\.",
                    r"os\.popen",
                    r"commands\.",
                    r"popen2\.",
                    r"os\.spawn",
                    r"exec[lv][ep]"
                ],
                "severity": "high"
            },
            "code_execution": {
                "patterns": [
                    r"eval\(",
                    r"exec\(",
                    r"execfile\(",
                    r"compile\(",
                    r"__import__"
                ],
                "severity": "critical"
            },
            "file_operations": {
                "patterns": [
                    r"open\(",
                    r"file\(",
                    r"os\.open",
                    r"os\.mkdir",
                    r"os\.makedirs",
                    r"os\.remove",
                    r"os\.unlink"
                ],
                "severity": "medium"
            }
        }
    
    def check_security(self, filepath: str) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        return {
            "pattern_analysis": self._check_vulnerability_patterns(content),
            "ast_analysis": self._analyze_ast(content),
            "bandit_analysis": self._run_bandit_scan(filepath),
            "input_validation": self._check_input_validation(content),
            "security_best_practices": self._check_security_best_practices(content)
        }
    
    def _check_vulnerability_patterns(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check for known vulnerability patterns"""
        findings = {}
        
        for vuln_type, info in self.known_vulnerabilities.items():
            findings[vuln_type] = []
            for pattern in info["patterns"]:
                matches = re.finditer(pattern, content)
                for match in matches:
                    findings[vuln_type].append({
                        "pattern": pattern,
                        "line": content[:match.start()].count('\n') + 1,
                        "severity": info["severity"],
                        "match": match.group()
                    })
        
        return findings
    
    def _analyze_ast(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Perform AST-based security analysis"""
        issues = {
            "dangerous_calls": [],
            "global_variables": [],
            "unsafe_attributes": []
        }
        
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'input']:
                            issues["dangerous_calls"].append({
                                "type": node.func.id,
                                "line": node.lineno,
                                "col": node.col_offset
                            })
                    self.generic_visit(node)
                
                def visit_Global(self, node):
                    for name in node.names:
                        issues["global_variables"].append({
                            "name": name,
                            "line": node.lineno
                        })
                    self.generic_visit(node)
                
                def visit_Attribute(self, node):
                    if isinstance(node.value, ast.Name):
                        if node.value.id == 'os' and node.attr in ['system', 'popen']:
                            issues["unsafe_attributes"].append({
                                "attribute": f"os.{node.attr}",
                                "line": node.lineno
                            })
                    self.generic_visit(node)
            
            SecurityVisitor().visit(tree)
        except SyntaxError:
            pass
        
        return issues
    
    def _run_bandit_scan(self, filepath: str) -> Dict[str, Any]:
        """Run Bandit security scanner"""
        b_mgr = b_manager.BanditManager()
        b_mgr.discover_files([filepath])
        b_mgr.run_tests()
        
        return {
            "issues": [
                {
                    "test_id": issue.test_id,
                    "issue_text": issue.text,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "line": issue.lineno
                }
                for issue in b_mgr.get_issue_list()
            ],
            "metrics": b_mgr.metrics.data
        }
    
    def _check_input_validation(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check for proper input validation"""
        issues = {
            "unvalidated_inputs": [],
            "type_hints_missing": []
        }
        
        try:
            tree = ast.parse(content)
            
            class InputValidationVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    # Check for missing type hints
                    if not node.returns and not any(isinstance(a, ast.AnnAssign) for a in node.args.args):
                        issues["type_hints_missing"].append({
                            "function": node.name,
                            "line": node.lineno
                        })
                    
                    # Look for input usage without validation
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if (isinstance(child.func, ast.Name) and 
                                child.func.id in ['input', 'raw_input']):
                                issues["unvalidated_inputs"].append({
                                    "function": node.name,
                                    "line": child.lineno
                                })
                    
                    self.generic_visit(node)
            
            InputValidationVisitor().visit(tree)
        except SyntaxError:
            pass
        
        return issues
    
    def _check_security_best_practices(self, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Check adherence to security best practices"""
        issues = {
            "hardcoded_secrets": [],
            "insecure_permissions": [],
            "unsafe_deserialization": []
        }
        
        # Check for potential hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]"
        ]
        
        for pattern in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues["hardcoded_secrets"].append({
                    "pattern": pattern,
                    "line": content[:match.start()].count('\n') + 1,
                    "match": match.group()
                })
        
        # Check for unsafe deserialization
        deser_patterns = [
            r"pickle\.loads?",
            r"yaml\.loads?",
            r"marshal\.loads?"
        ]
        
        for pattern in deser_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                issues["unsafe_deserialization"].append({
                    "pattern": pattern,
                    "line": content[:match.start()].count('\n') + 1,
                    "match": match.group()
                })
        
        return issues 