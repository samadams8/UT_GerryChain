import os
import json
import pandas as pd
import yaml
from typing import Dict, Any, List, Optional, Union

def _flatten_value(value: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Recursively flatten a value (dict, list, tuple, or scalar) into a flat dictionary.
    
    Parameters
    ----------
    value : Any
        The value to flatten. Can be a dict, list, tuple, or scalar.
    prefix : str, optional
        Prefix to prepend to all keys in the flattened dictionary.
        Default is empty string.
    
    Returns
    -------
    Dict[str, Any]
        A flat dictionary with keys like "prefix_key" or "prefix_0" for nested structures.
    
    Examples
    --------
    >>> _flatten_value({"0": 1, "1": 2}, "population")
    {'population_0': 1, 'population_1': 2}
    
    >>> _flatten_value([1, 2, 3], "values")
    {'values_0': 1, 'values_1': 2, 'values_2': 3}
    
    >>> _flatten_value({"a": {"b": 1}}, "data")
    {'data_a_b': 1}
    
    >>> _flatten_value([{"x": 1}, {"y": 2}], "items")
    {'items_0_x': 1, 'items_1_y': 2}
    """
    result = {}
    
    if isinstance(value, dict):
        if not value:  # Empty dict
            return result
        for key, sub_value in value.items():
            new_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened = _flatten_value(sub_value, new_prefix)
            result.update(flattened)
    elif isinstance(value, (list, tuple)):
        if not value:  # Empty list/tuple
            return result
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}_{idx}" if prefix else str(idx)
            flattened = _flatten_value(item, new_prefix)
            result.update(flattened)
    else:
        # Scalar value
        if prefix:
            result[prefix] = value
        else:
            # This shouldn't happen in normal usage, but handle it
            result[str(value)] = value
    
    return result


def read_jsonl_table(filepath: str, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Read a JSONL file and return a pandas DataFrame with flattened nested structures.
    
    This function reads a JSONL (JSON Lines) file where each line is a JSON object,
    extracts only the specified columns, and recursively flattens nested dictionaries
    and lists/tuples into separate columns.
    
    Parameters
    ----------
    filepath : str
        Path to the JSONL file to read.
    columns : str or List[str]
        Column name(s) to extract from the JSONL file. Required parameter.
        If a single string is provided, it will be converted to a list.
        The function will raise ValueError if this parameter is None or not provided.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with flattened columns. Dictionary keys and list/tuple
        indices are appended to the root column name with underscores.
        For example, {"population": {"0": 9000}} becomes a column "population_0".
    
    Raises
    ------
    ValueError
        If columns parameter is None or not provided.
    FileNotFoundError
        If the specified filepath does not exist.
    
    Examples
    --------
    >>> df = read_jsonl_table("output.jsonl", "population")
    >>> # Returns DataFrame with columns like population_0, population_1, etc.
    
    >>> df = read_jsonl_table("output.jsonl", ["population", "polsby_popper"])
    >>> # Returns DataFrame with flattened columns for both specified keys
    """
    # Validate columns parameter
    if columns is None:
        raise ValueError("columns parameter is required and cannot be None")
    
    # Normalize columns to a list
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list):
        raise ValueError(f"columns must be a string or list of strings, got {type(columns)}")
    
    if not columns:
        raise ValueError("columns parameter cannot be an empty list")
    
    # Read and parse JSONL file
    flattened_rows = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            try:
                row_data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {filepath}: {e}")
            
            # Extract only the requested columns
            extracted_data = {}
            for col in columns:
                if col in row_data:
                    extracted_data[col] = row_data[col]
                # If column doesn't exist, we'll add it as None/empty later
            
            # Flatten the extracted data
            flattened_row = {}
            for col in columns:
                if col in extracted_data:
                    flattened = _flatten_value(extracted_data[col], col)
                    flattened_row.update(flattened)
                # If column is missing, we don't add any flattened columns for it
                # pandas will handle missing columns with NaN
            
            flattened_rows.append(flattened_row)
    
    # Convert to DataFrame
    if not flattened_rows:
        # Return empty DataFrame with appropriate structure
        return pd.DataFrame()
    
    df = pd.DataFrame(flattened_rows)
    
    return df


def sort_subentries(df: pd.DataFrame, sort_column: str, ascending: bool = True) -> pd.DataFrame:
    """
    Sort sub-entries (e.g., district indices) within each row by a specified metric,
    and reorder all corresponding sub-entries across all columns to maintain consistency.
    
    This function is useful when you have columns like `population_0`, `population_1`,
    `polsby_popper_0`, `polsby_popper_1`, etc., and you want to sort the districts
    (indices 0, 1, 2, ...) by one metric (e.g., population) and have all other
    metrics follow that ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns that have sub-entry suffixes (e.g., "base_0", "base_1").
    sort_column : str
        Name of the column to sort by. Must be a column that exists in the DataFrame
        and has sub-entry suffixes (e.g., "population_0", "population_1", ...).
    ascending : bool, optional
        Whether to sort in ascending order. Default is True.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with sub-entries sorted and reordered. Columns that don't
        follow the base_suffix pattern are left unchanged.
    
    Raises
    ------
    ValueError
        If sort_column doesn't exist in the DataFrame or has no sub-entry columns.
    
    Examples
    --------
    >>> df = read_jsonl_table("output.jsonl", ["population", "polsby_popper"])
    >>> # Original: population_0=818232, population_1=817463, ...
    >>> df_sorted = sort_subentries(df, "population", ascending=True)
    >>> # After: population_0=817463 (smallest), population_1=817944, ...
    >>> # All polsby_popper columns are also reordered to match
    """
    import re
    import numpy as np
    
    # Validate input
    if df.empty:
        return df.copy()
    
    # Parse column names to identify base names and suffixes
    # Pattern: base_name_suffix where suffix is typically _0, _1, _2, etc.
    # But could also be _a, _b, etc. or nested like _a_b
    # We'll look for columns that match the sort_column base name
    # The sort_column can be either a base name (e.g., "population") or a full column name (e.g., "population_0")
    # If it's a full column name, extract the base name
    if sort_column in df.columns:
        # It's a full column name, extract base name
        # Try to match pattern base_suffix
        match = re.match(r"^(.+)_(.+)$", sort_column)
        if match:
            base_name = match.group(1)
            sort_column = base_name  # Use base name for pattern matching
        else:
            # Column exists but doesn't follow pattern - can't sort sub-entries
            raise ValueError(
                f"Sort column '{sort_column}' exists but has no sub-entry pattern. "
                f"Expected columns like '{sort_column}_0', '{sort_column}_1', etc."
            )
    
    base_pattern = re.escape(sort_column)
    sort_col_pattern = re.compile(f"^{base_pattern}_(.+)$")
    
    # Find all columns that match the sort column pattern
    sort_cols = [col for col in df.columns if sort_col_pattern.match(col)]
    
    if not sort_cols:
        raise ValueError(
            f"Sort column '{sort_column}' has no sub-entry columns. "
            f"Expected columns like '{sort_column}_0', '{sort_column}_1', etc."
        )
    
    # Extract suffixes from sort columns
    # For each sort column, extract the suffix (everything after the last underscore)
    # But we need to be careful - the suffix might be nested like "_0_1"
    # We'll extract the full suffix after the base name
    suffix_to_sort_col = {}
    for col in sort_cols:
        match = sort_col_pattern.match(col)
        if match:
            suffix = match.group(1)  # Everything after "base_"
            suffix_to_sort_col[suffix] = col
    
    if not suffix_to_sort_col:
        raise ValueError(f"Could not parse suffixes from sort column '{sort_column}'")
    
    # Find all columns that have matching suffixes (across all base names)
    # Pattern: any_base_name_suffix where suffix matches one of our suffixes
    escaped_suffixes = '|'.join(re.escape(s) for s in suffix_to_sort_col.keys())
    suffix_pattern = re.compile(f"^(.+)_({escaped_suffixes})$")
    
    columns_to_reorder = {}
    unchanged_columns = []
    
    for col in df.columns:
        match = suffix_pattern.match(col)
        if match:
            base_name = match.group(1)
            suffix = match.group(2)
            if base_name not in columns_to_reorder:
                columns_to_reorder[base_name] = {}
            columns_to_reorder[base_name][suffix] = col
        else:
            # Column doesn't follow the pattern, leave it unchanged
            unchanged_columns.append(col)
    
    if not columns_to_reorder:
        # No columns to reorder, return copy
        return df.copy()
    
    # Get all unique suffixes across all rows (to determine maximum number of sub-entries)
    all_suffixes = set()
    for idx in df.index:
        for suffix in suffix_to_sort_col.keys():
            col_name = suffix_to_sort_col[suffix]
            if not pd.isna(df.loc[idx, col_name]):
                all_suffixes.add(suffix)
    
    # Sort suffixes to get a canonical order (we'll use this for new column names)
    canonical_suffixes = sorted(all_suffixes, key=lambda s: (len(s), s))
    num_suffixes = len(canonical_suffixes)
    
    # Create new column names for each base
    new_columns = {}
    for base_name in columns_to_reorder.keys():
        for new_pos in range(num_suffixes):
            new_col_name = f"{base_name}_{new_pos}"
            new_columns[new_col_name] = None  # Will be filled
    
    # Initialize result DataFrame with new structure
    result_data = {}
    for col in unchanged_columns:
        result_data[col] = df[col].values
    
    # Initialize new columns with NaN
    for base_name in columns_to_reorder.keys():
        for new_pos in range(num_suffixes):
            new_col_name = f"{base_name}_{new_pos}"
            result_data[new_col_name] = np.full(len(df), np.nan)
    
    # For each row, sort and assign values
    for idx in df.index:
        # Get sort values for this row
        sort_values = {}
        available_suffixes = []
        for suffix, col_name in suffix_to_sort_col.items():
            value = df.loc[idx, col_name]
            if not pd.isna(value):
                sort_values[suffix] = value
                available_suffixes.append(suffix)
        
        # Sort available suffixes by their values
        sorted_suffixes = sorted(
            available_suffixes,
            key=lambda s: sort_values[s],
            reverse=not ascending
        )
        
        # Assign values to new positions
        for new_pos, old_suffix in enumerate(sorted_suffixes):
            # Assign value from old_suffix to new position for all base names
            for base_name, suffix_to_col in columns_to_reorder.items():
                if old_suffix in suffix_to_col:
                    old_col_name = suffix_to_col[old_suffix]
                    new_col_name = f"{base_name}_{new_pos}"
                    result_data[new_col_name][idx] = df.loc[idx, old_col_name]
    
    # Create result DataFrame
    result_df = pd.DataFrame(result_data, index=df.index)
    
    # Reorder columns to match original order as much as possible
    # Put unchanged columns first, then new columns grouped by base name
    new_column_order = unchanged_columns.copy()
    for base_name in sorted(columns_to_reorder.keys()):
        for new_pos in range(num_suffixes):
            new_col_name = f"{base_name}_{new_pos}"
            if new_col_name in result_df.columns:
                new_column_order.append(new_col_name)
    
    # Only include columns that actually exist
    new_column_order = [col for col in new_column_order if col in result_df.columns]
    result_df = result_df[new_column_order]
    
    return result_df
