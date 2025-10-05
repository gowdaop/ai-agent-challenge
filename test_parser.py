"""
test_parser.py - Automated tests for generated bank statement parsers

Usage:
    pytest test_parser.py                    # Test all available parsers
    pytest test_parser.py::test_icici_parser # Test specific bank
    pytest test_parser.py -v                 # Verbose mode
"""

import pytest
import pandas as pd
import importlib.util
from pathlib import Path
from typing import Callable


def load_parser(bank_name: str) -> Callable:
    """
    Dynamically load parser module for a given bank
    
    Args:
        bank_name: Bank identifier (e.g., 'icici', 'sbi')
        
    Returns:
        parse function from the module
        
    Raises:
        FileNotFoundError: If parser doesn't exist
        AttributeError: If parser missing parse() function
    """
    parser_path = Path("custom_parsers") / f"{bank_name}_parser.py"
    
    if not parser_path.exists():
        raise FileNotFoundError(
            f"Parser not found: {parser_path}\n"
            f"Run: python agent.py --target {bank_name}"
        )
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"{bank_name}_parser", 
        parser_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'parse'):
        raise AttributeError(
            f"Parser {parser_path} missing parse() function"
        )
    
    return module.parse


def get_test_data(bank_name: str) -> tuple[Path, pd.DataFrame]:
    """
    Get test PDF path and expected DataFrame for a bank
    
    Args:
        bank_name: Bank identifier
        
    Returns:
        Tuple of (pdf_path, expected_dataframe)
    """
    pdf_path = Path("data") / bank_name / f"{bank_name}_sample.pdf"
    csv_path = Path("data") / bank_name / "result.csv"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Test PDF not found: {pdf_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
    
    expected_df = pd.read_csv(csv_path)
    
    return pdf_path, expected_df


def validate_dataframe_schema(df: pd.DataFrame) -> list[str]:
    """
    Validate DataFrame has required schema
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    
    # Check columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check column order
    if list(df.columns) != required_columns:
        errors.append(
            f"Column order incorrect.\n"
            f"  Expected: {required_columns}\n"
            f"  Got: {list(df.columns)}"
        )
    
    # Check numeric columns
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
    
    # Check string columns
    for col in ['Date', 'Description']:
        if col in df.columns:
            if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                errors.append(f"Column '{col}' should be string-like, got {df[col].dtype}")
    
    return errors


def compare_dataframes(result: pd.DataFrame, expected: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Compare result DataFrame with expected, providing detailed feedback
    
    Returns:
        Tuple of (is_equal, list_of_differences)
    """
    differences = []
    
    # Row count check
    if len(result) != len(expected):
        differences.append(
            f"Row count mismatch: got {len(result)} rows, expected {len(expected)}"
        )
    
    # Column check
    if list(result.columns) != list(expected.columns):
        differences.append(
            f"Column mismatch:\n"
            f"  Got: {list(result.columns)}\n"
            f"  Expected: {list(expected.columns)}"
        )
        return False, differences
    
    # Value comparison for each column
    min_rows = min(len(result), len(expected))
    
    for col in expected.columns:
        # Compare values
        result_col = result[col].head(min_rows).reset_index(drop=True)
        expected_col = expected[col].head(min_rows).reset_index(drop=True)
        
        if pd.api.types.is_numeric_dtype(expected_col):
            # Numeric comparison with NaN handling
            # Two NaNs are considered equal for our purposes
            both_nan = pd.isna(result_col) & pd.isna(expected_col)
            values_equal = result_col == expected_col
            mismatches = ~(both_nan | values_equal)
        else:
            # String comparison (NaN-safe)
            both_nan = pd.isna(result_col) & pd.isna(expected_col)
            values_equal = result_col.astype(str) == expected_col.astype(str)
            mismatches = ~(both_nan | values_equal)
        
        if mismatches.any():
            mismatch_count = mismatches.sum()
            differences.append(
                f"Column '{col}': {mismatch_count} mismatched values"
            )
            
            # Show first 3 mismatches
            mismatch_indices = mismatches[mismatches].index[:3]
            for idx in mismatch_indices:
                differences.append(
                    f"  Row {idx}: got '{result_col.iloc[idx]}', "
                    f"expected '{expected_col.iloc[idx]}'"
                )
    
    is_equal = len(differences) == 0
    return is_equal, differences


# Parametrized test that discovers all available banks
def discover_banks() -> list[str]:
    """Discover all banks with test data"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    banks = []
    for bank_dir in data_dir.iterdir():
        if bank_dir.is_dir():
            pdf_file = bank_dir / f"{bank_dir.name}_sample.pdf"
            csv_file = bank_dir / "result.csv"
            if pdf_file.exists() and csv_file.exists():
                banks.append(bank_dir.name)
    
    return banks


@pytest.mark.parametrize("bank_name", discover_banks())
def test_bank_parser(bank_name: str):
    """
    Test parser for a specific bank
    
    This test:
    1. Loads the generated parser
    2. Runs it on the sample PDF
    3. Validates the schema
    4. Compares output with expected CSV
    """
    # Load parser
    parse_func = load_parser(bank_name)
    
    # Get test data
    pdf_path, expected_df = get_test_data(bank_name)
    
    # Execute parser
    result_df = parse_func(str(pdf_path))
    
    # Validate it returns a DataFrame
    assert isinstance(result_df, pd.DataFrame), \
        f"parse() must return pd.DataFrame, got {type(result_df)}"
    
    # Validate schema
    schema_errors = validate_dataframe_schema(result_df)
    assert not schema_errors, \
        f"Schema validation failed:\n" + "\n".join(f"  - {e}" for e in schema_errors)
    
    # Compare with expected
    is_equal, differences = compare_dataframes(result_df, expected_df)
    
    assert is_equal, \
        f"Parser output doesn't match expected CSV:\n" + \
        "\n".join(f"  - {d}" for d in differences)
    
    print(f"✓ {bank_name.upper()} parser: {len(result_df)} transactions extracted correctly")


# Individual bank tests for easier debugging
def test_icici_parser():
    """Test ICICI bank parser specifically"""
    test_bank_parser("icici")


def test_sbi_parser():
    """Test SBI bank parser (if available)"""
    if "sbi" not in discover_banks():
        pytest.skip("SBI parser not generated yet")
    test_bank_parser("sbi")


def test_hdfc_parser():
    """Test HDFC bank parser (if available)"""
    if "hdfc" not in discover_banks():
        pytest.skip("HDFC parser not generated yet")
    test_bank_parser("hdfc")


# Smoke test - verify agent can be imported
def test_agent_imports():
    """Verify agent.py can be imported without errors"""
    try:
        import agent
        assert hasattr(agent, 'EnhancedParserAgent'), \
            "agent.py missing EnhancedParserAgent class"
        assert hasattr(agent, 'main'), \
            "agent.py missing main() function"
    except ImportError as e:
        pytest.fail(f"Failed to import agent.py: {e}")


# Integration test - verify file structure
def test_project_structure():
    """Verify required project files exist"""
    required_files = [
        "agent.py",
        "data/icici/icici_sample.pdf",
        "data/icici/result.csv",
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    
    assert not missing, \
        f"Missing required files:\n" + "\n".join(f"  - {f}" for f in missing)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])