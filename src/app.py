import json
from handler import diagnose_stress
import logging

def lambda_handler(event, context):
    logging.info(event)
    intent = event["sessionState"]["intent"]["name"]
    if intent == "DiagnoseStress":
        response = diagnose_stress(event)
    else:
        raise Exception(f"Intent {intent} does not exist")
    return response