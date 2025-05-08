from textwrap import dedent

import pandas as pd


def description(
    df: pd.DataFrame, description_strategy: str = "head", num_rows: int = 1
) -> str:
    """Returns a description of the given data for LLM"""

    if description_strategy == "head":
        return description_by_head(df, num_rows)
    elif description_strategy == "dtypes":
        return description_by_dtypes(df)
    else:
        raise ValueError(f"Unknown description_strategy: {description_strategy}")


def description_by_head(df: pd.DataFrame, num_rows: int = 5) -> str:
    
    
    print ("num_rows", num_rows)
    num_rows=int(df.shape[0]/3)
    print ("num_rows", num_rows)
    
    
    if len(df) < num_rows:
        head_part = str(df.to_markdown())
    else:
        head_part = str(df.sample(num_rows, random_state=0).to_markdown())
        
    print ("head_part", head_part)

    return dedent(
        f"""
        This is the result of `print(df.head())`:

        {head_part}
        """
    )


def description_by_dtypes(df: pd.DataFrame) -> str:
    return dedent(
        f"""
        This is the result of `print(df.dtypes)`:

        {str(df.dtypes.to_markdown())}
        """
    )
