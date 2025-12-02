# REF: https://github.com/JackKuo666/ClinicalTrials-MCP-Server
from mcp.server.fastmcp import FastMCP, Context
from pytrials.client import ClinicalTrials
import pandas as pd
import os
import logging
import sys

MAX_OUTPUT_CHARS = 20000

logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chembl_mcp")

try:
    mcp = FastMCP(
        name="clinicaltrial_tools",
    )
    logger.info("Clinical Trial MCP server initialized successfully")
except Exception as e:
    err_msg = f"Error: {str(e)}"
    logger.error(f"{err_msg}")

ct = ClinicalTrials()

# Helper functions
def load_csv_file(filename):
    """Load data from a CSV file"""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

def format_limited_output(df, max_rows=None, max_chars=MAX_OUTPUT_CHARS):
    """Format DataFrame output with character limit and metadata"""
    if df is None or df.empty:
        return "No data available"
    
    total_rows = len(df)
    
    # If maximum rows are specified, limit the output rows
    if max_rows and max_rows < total_rows:
        display_df = df.head(max_rows)
        rows_shown = max_rows
    else:
        display_df = df
        rows_shown = total_rows
    
    # Convert to string
    output = display_df.to_string()
    
    # If exceeding character limit, truncate
    if len(output) > max_chars:
        output = output[:max_chars] + "\n...[Output truncated]"
    
    # Add metadata
    metadata = f"\n\nData summary: Total {total_rows} records, showing {rows_shown} records."
    
    return output + metadata

def load_full_studies():
    """Load the full studies data from CSV"""
    return load_csv_file("full_studies.csv")

def list_available_csv_files():
    """List all available CSV files in the current directory"""
    return [f for f in os.listdir('.') if f.endswith('.csv')]

@mcp.resource("clinicaltrials://full_studies")
def get_full_studies_resource() -> str:
    """Get the full studies data as a resource"""
    df = load_full_studies()
    if df is not None:
        return format_limited_output(df)
    return "Full studies data not available"

@mcp.resource("clinicaltrials://csv/{filename}")
def get_csv_file(filename: str) -> str:
    """Get data from a specific CSV file"""
    # Ensure the filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    df = load_csv_file(filename)
    if df is not None:
        return format_limited_output(df)
    return f"CSV file {filename} not available"

@mcp.resource("clinicaltrials://available_files")
def get_available_files() -> str:
    """Get a list of all available CSV files"""
    files = list_available_csv_files()
    if files:
        return "\n".join(files)
    return "No CSV files available"

@mcp.resource("clinicaltrials://study/{nct_id}")
def get_study_by_id(nct_id: str) -> str:
    """Get a specific study by NCT ID"""
    df = load_full_studies()
    if df is not None and "NCT Number" in df.columns:
        study = df[df["NCT Number"] == nct_id]
        if not study.empty:
            return format_limited_output(study)
    
    # If not found in local data, try to fetch from API
    try:
        study = ct.get_study_fields(
            search_expr=f"NCT Number={nct_id}",
            fields=["NCT Number", "Conditions", "Study Title", "Brief Summary", "Detailed Description"],
            max_studies=1
        )
        if len(study) > 1:  # Header + data
            return pd.DataFrame.from_records(study[1:], columns=study[0]).to_string()
    except Exception as e:
        return f"Error fetching study: {str(e)}"
    
    return f"Study with NCT ID {nct_id} not found"

# Tools
@mcp.tool()
def search_clinical_trials_and_save_studies_to_csv(search_expr: str, max_studies: int = 10, save_csv: bool = True, filename: str = "corona_fields.csv", fields: list = None) -> str:
    """
    Search for clinical trials using a search expression and save the results to a CSV file
    
    Args:
        search_expr: Search expression (e.g., "Coronavirus+COVID")
        max_studies: Maximum number of studies to return (default: 10)
        save_csv: Whether to save the results as a CSV file (default: False)
        filename: Name of the CSV file to save (default: search_results.csv)
        fields: List of fields to include (default: NCT Number, Conditions, Study Title, Brief Summary)
    
    Returns:
        String representation of the search results
    """
    try:
        # Default fields if none provided
        if fields is None:
            fields = ["NCT Number", "Conditions", "Study Title", "Brief Summary"]
        
        # Get study fields
        fmt = "csv" if save_csv else None
        results = ct.get_study_fields(
            search_expr=search_expr,
            fields=fields,
            max_studies=max_studies,
            fmt=fmt
        )
        
        if len(results) > 1:  # Header + data
            df = pd.DataFrame.from_records(results[1:], columns=results[0])
            
            # Save to CSV if requested
            if save_csv:
                csv_filename = filename or f"search_results_{search_expr.replace('+', '_')}.csv"
                csv_filename = os.path.basename(csv_filename)
                df.to_csv(csv_filename, index=False)
                storage_info = f"Complete results have been saved to file {csv_filename}"
                return f"Results saved to {csv_filename}\n\n{format_limited_output(df)}\n{storage_info}"
            
            return format_limited_output(df)
        return "No results found"
    except Exception as e:
        return f"Error searching clinical trials: {str(e)}"

@mcp.tool()
def get_full_study_details(nct_id: str) -> str:
    """
    Get detailed information about a specific clinical trial
    
    Args:
        nct_id: The NCT ID of the clinical trial
    
    Returns:
        String representation of the study details
    """
    try:
        study = ct.get_full_studies(search_expr=f"NCT Number={nct_id}", max_studies=1)
        if len(study) > 1:  # Header + data
            df = pd.DataFrame.from_records(study[1:], columns=study[0])
            return format_limited_output(df)
        return f"Study with NCT ID {nct_id} not found"
    except Exception as e:
        return f"Error fetching study details: {str(e)}"

@mcp.tool()
def get_studies_by_keyword(keyword: str, max_studies: int = 20, save_csv: bool = True, filename: str = None) -> str:
    """
    Get studies related to a specific keyword
    
    Args:
        keyword: Keyword to search for
        max_studies: Maximum number of studies to return (default: 20)
        save_csv: Whether to save the results as a CSV file (default: False)
        filename: Name of the CSV file to save (default: keyword_results.csv)
    
    Returns:
        String representation of the studies
    """
    try:
        fields = ["NCT Number", "Conditions", "Study Title", "Brief Summary"]
        results = ct.get_study_fields(
            search_expr=keyword,
            fields=fields,
            max_studies=max_studies
        )
        
        if len(results) > 1:  # Header + data
            df = pd.DataFrame.from_records(results[1:], columns=results[0])
            
            # Save to CSV if requested
            if save_csv:
                csv_filename = filename or f"keyword_results_{keyword.replace(' ', '_')}.csv"
                csv_filename = os.path.basename(csv_filename)
                df.to_csv(csv_filename, index=False)
                storage_info = f"Complete results have been saved to file {csv_filename}"
                return f"Results saved to {csv_filename}\n\n{format_limited_output(df)}\n{storage_info}"
                
            return format_limited_output(df)
        return f"No studies found for keyword: {keyword}"
    except Exception as e:
        return f"Error searching studies by keyword: {str(e)}"

@mcp.tool()
def get_full_studies_and_save(search_expr: str, max_studies: int = 20, filename: str = "full_studies.csv") -> str:
    """
    Get full studies data and save to CSV
    
    Args:
        search_expr: Search expression (e.g., "Coronavirus+COVID")
        max_studies: Maximum number of studies to return (default: 20)
        filename: Name of the CSV file to save (default: full_studies.csv)
    
    Returns:
        Message indicating the results were saved
    """
    try:
        # Get full studies
        full_studies = ct.get_full_studies(search_expr=search_expr, max_studies=max_studies)
        
        if len(full_studies) > 1:  # Header + data
            # Convert to DataFrame
            df = pd.DataFrame.from_records(full_studies[1:], columns=full_studies[0])

            # Save to CSV
            safe_filename = os.path.basename(filename)
            df.to_csv(safe_filename, index=False)

            return f"Successfully saved {len(df)} full studies to {safe_filename}"
        return "No results found to save"
    except Exception as e:
        return f"Error saving full studies to CSV: {str(e)}"

@mcp.tool()
def load_csv_data(filename: str) -> str:
    """
    Load and display data from a CSV file
    
    Args:
        filename: Name of the CSV file to load
    
    Returns:
        String representation of the CSV data
    """
    # Ensure the filename ends with .csv
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    df = load_csv_file(filename)
    if df is not None:
        return f"Loaded data from {filename}:\n\n{format_limited_output(df)}"
    return f"CSV file {filename} not found or could not be loaded"

@mcp.tool()
def list_saved_csv_files() -> str:
    """
    List all available CSV files in the current directory
    
    Returns:
        String representation of the available CSV files
    """
    files = list_available_csv_files()
    if files:
        return f"Available CSV files:\n\n{chr(10).join(files)}"
    return "No CSV files available"

# Prompts
@mcp.prompt()
def search_trials_prompt() -> str:
    """Prompt for searching clinical trials"""
    return """
    I can help you search for clinical trials. Please provide:
    
    1. A search term or condition (e.g., "COVID-19", "Diabetes", "Cancer")
    2. Optional: Maximum number of results to return
    
    I'll search the ClinicalTrials.gov database and return relevant studies.
    """

@mcp.prompt()
def analyze_trial_prompt() -> str:
    """Prompt for analyzing a specific clinical trial"""
    return """
    I can help you analyze a specific clinical trial. Please provide:
    
    1. The NCT ID of the trial (e.g., NCT04280705)
    
    I'll retrieve detailed information about the trial and provide an analysis.
    """

@mcp.prompt()
def csv_management_prompt() -> str:
    """Prompt for managing CSV files"""
    return """
    I can help you manage your clinical trials CSV files. I can:
    
    1. List all available CSV files
    2. Load and display data from a specific CSV file
    3. Save new search results to CSV files
    
    What would you like to do?
    """

if __name__ == "__main__":
    mcp.run()