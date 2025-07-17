from mcp.server.fastmcp import FastMCP
import logging
import sys
from typing import Any, List, Dict
from chembl_webresource_client.new_client import new_client

MAXIMUM_ACTIVITY = 100

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
        name="chembl_tools",
    )
    logger.info("ChEMBL MCP server initialized successfully")
except Exception as e:
    err_msg = f"Error: {str(e)}"
    logger.error(f"{err_msg}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@mcp.tool()
async def compount_activity(compound_name: str) -> List[Dict[str, Any]]:
    """activity data for the specified compound
    
    Args:
        compound_name: name of compound
        
    Returns:
        List of activity data
    """
    client = new_client
    molecule_id = client.molecule.filter(pref_name__iexact=compound_name).only('molecule_chembl_id')[0]
    # TODO: consider other types of activities
    activity = list(client.activity.filter(molecule_chembl_id=molecule_id['molecule_chembl_id']).filter(standard_type="IC50").only(['pchembl_value', 'assay_description', 'canonical_smiles']))
    if len(activity)>MAXIMUM_ACTIVITY:
        activity=activity[:MAXIMUM_ACTIVITY] # TODO: consider longer context
    return activity

@mcp.tool()
async def target_activity(target_name: str) -> List[Dict[str, Any]]:
    """activity data for the specified target
    
    Args:
        target_name: name of target
        
    Returns:
        List of activity data
    """
    client = new_client
    target_id = client.target.filter(target_synonym__icontains=target_name, organism='Homo sapiens').only('target_chembl_id')[0]
    # TODO: consider other types of activities
    activity = list(client.activity.filter(target_chembl_id=target_id['target_chembl_id']).filter(standard_type="IC50").only(['pchembl_value', 'assay_description', 'canonical_smiles']))
    if len(activity)>MAXIMUM_ACTIVITY:
        activity=activity[:MAXIMUM_ACTIVITY] # TODO: consider longer context
    return activity

if __name__ == "__main__":
    mcp.run()

