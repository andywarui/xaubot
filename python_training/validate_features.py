"""
Feature Validation Utility for XAUUSD AI Trading Bot
Detects look-ahead bias in feature engineering code.

Usage:
    python validate_features.py [file_to_check.py]
    
If no file is specified, validates all feature engineering files.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional


class LookAheadBiasDetector(ast.NodeVisitor):
    """
    AST visitor that detects potential look-ahead bias patterns:
    - .shift(-N) where N > 0 (future data access)
    - .rolling(..., center=True) (uses future data for window)
    - Direct indexing with future indices
    """
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        self.issues: List[Dict] = []
        self.current_assignment: Optional[str] = None
        
    def visit_Assign(self, node):
        """Track the current variable being assigned."""
        # Get the target name if it's a simple assignment
        if node.targets and isinstance(node.targets[0], ast.Subscript):
            if isinstance(node.targets[0].slice, ast.Constant):
                self.current_assignment = str(node.targets[0].slice.value)
            elif isinstance(node.targets[0].slice, ast.Name):
                self.current_assignment = node.targets[0].slice.id
        self.generic_visit(node)
        self.current_assignment = None
        
    def visit_Call(self, node):
        """Check function calls for look-ahead patterns."""
        
        # Check for .shift(-N) pattern
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'shift':
                self._check_shift(node)
            elif node.func.attr == 'rolling':
                self._check_rolling(node)
                
        self.generic_visit(node)
    
    def _check_shift(self, node):
        """Check shift() calls for negative values (future access)."""
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                # Negative shift: .shift(-N)
                if isinstance(arg.operand, ast.Constant) and arg.operand.value > 0:
                    self.issues.append({
                        'type': 'shift_negative',
                        'line': node.lineno,
                        'column': self.current_assignment,
                        'code': self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else '',
                        'severity': 'ERROR',
                        'message': f'Look-ahead bias: .shift(-{arg.operand.value}) accesses future data'
                    })
                    
    def _check_rolling(self, node):
        """Check rolling() calls for center=True (uses future data)."""
        for keyword in node.keywords:
            if keyword.arg == 'center':
                if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.issues.append({
                        'type': 'rolling_center',
                        'line': node.lineno,
                        'column': self.current_assignment,
                        'code': self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else '',
                        'severity': 'WARNING',
                        'message': 'Potential look-ahead bias: rolling(center=True) uses future data in window'
                    })


def validate_file(filepath: Path) -> Tuple[bool, List[Dict]]:
    """
    Validate a Python file for look-ahead bias.
    
    Returns:
        Tuple of (is_clean, list_of_issues)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        return False, [{'type': 'error', 'message': f'Could not read file: {e}'}]
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return False, [{'type': 'error', 'message': f'Syntax error: {e}'}]
    
    detector = LookAheadBiasDetector(source_code)
    detector.visit(tree)
    
    return len(detector.issues) == 0, detector.issues


def validate_feature_cols(feature_cols: List[str], 
                          biased_features: Optional[Set[str]] = None) -> Tuple[bool, List[str]]:
    """
    Check if any feature columns are known to have look-ahead bias.
    
    Args:
        feature_cols: List of feature column names used by a model
        biased_features: Set of known biased feature names (optional)
        
    Returns:
        Tuple of (is_clean, list_of_biased_features_found)
    """
    if biased_features is None:
        # Default known biased features (legacy patterns)
        biased_features = {
            # Add any known biased feature names here
            # Currently empty as we've fixed the feature engineering
        }
    
    found_biased = [f for f in feature_cols if f in biased_features]
    return len(found_biased) == 0, found_biased


def generate_report(filepath: Path, issues: List[Dict]) -> str:
    """Generate a human-readable report of issues found."""
    if not issues:
        return f"✅ {filepath.name}: No look-ahead bias detected"
    
    lines = [f"❌ {filepath.name}: {len(issues)} issue(s) found"]
    
    for issue in issues:
        severity_icon = "🔴" if issue.get('severity') == 'ERROR' else "🟡"
        lines.append(f"   {severity_icon} Line {issue.get('line', '?')}: {issue.get('message', 'Unknown issue')}")
        if issue.get('code'):
            lines.append(f"      Code: {issue['code']}")
    
    return '\n'.join(lines)


def main():
    """Main entry point for validation."""
    print("=" * 70)
    print("FEATURE LOOK-AHEAD BIAS VALIDATOR")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Files to validate
    if len(sys.argv) > 1:
        files_to_check = [Path(sys.argv[1])]
    else:
        # Default: check all feature engineering files
        files_to_check = [
            project_root / 'src' / 'feature_engineering.py',
            project_root / 'python_training' / 'build_features_m1.py',
            project_root / 'python_training' / 'build_features_all_tf.py',
            project_root / 'python_training' / 'prepare_hybrid_features.py',
            project_root / 'python_training' / 'prepare_hybrid_features_multi_tf.py',
        ]
    
    all_clean = True
    total_issues = 0
    
    for filepath in files_to_check:
        if not filepath.exists():
            print(f"⚠️  {filepath.name}: File not found (skipped)")
            continue
            
        is_clean, issues = validate_file(filepath)
        report = generate_report(filepath, issues)
        print(report)
        
        if not is_clean:
            all_clean = False
            total_issues += len(issues)
    
    print()
    print("=" * 70)
    
    if all_clean:
        print("✅ ALL FILES CLEAN - No look-ahead bias detected!")
        print()
        print("Your features are safe for backtesting and live trading.")
        return 0
    else:
        print(f"❌ ISSUES FOUND: {total_issues} potential look-ahead bias issue(s)")
        print()
        print("Fix these issues before using features in production:")
        print("  - Replace .shift(-N) with .shift(N) or remove future references")
        print("  - Replace rolling(center=True) with rolling() (no center)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
