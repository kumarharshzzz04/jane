"""Bind executed rows into model-proposed Vega-Lite specs."""



def _bind_chart_data(chart: dict, columns: list[str], rows: list[list]) -> dict:
    """Bind actual query rows into a Vega-Lite spec — the model never supplies data."""
    values = [dict(zip(columns, row)) for row in rows[:200]]
    bound = dict(chart)
    bound["data"] = {"values": values}
    return bound

