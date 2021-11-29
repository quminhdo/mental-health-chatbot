from classifier import StressClassifier
import logging

stress_classifier = StressClassifier("models/bert-base-uncased", "models/stress-diagnosis-model.pth")
logging.info('Loaded model successfully')

def diagnose_stress(event):
    response = {}
    text = event["inputTranscript"]
    activeContexts = [] if event["sessionState"].get("activeContexts") is None else event["sessionState"]["activeContexts"]
    sessionAttributes = {} if event["sessionState"].get("sessionAttributes") is None else event["sessionState"]["sessionAttributes"]
    logging.info('Prediction starts')
    is_stress = stress_classifier.predict(text)
    logging.info('Prediction ends')
    if is_stress:
        dialogAction = {
            "type":"ConfirmIntent"
        }
        intent = {
            "name": "MakeAppointment",
            "slots": event["sessionState"]["intent"]["slots"],
            "state": "InProgress"
        }
        messages = [
            {
                "contentType": "PlainText",
                "content": "It seems that you are stressed. You should see a therapist asap. Do you want to make an appointment?"
            }
        ]
    else:
        dialogAction = {
            "type":"Close"
        }
        intent = {
            "name": event["sessionState"]["intent"]["name"],
            "slots": event["sessionState"]["intent"]["slots"],
            "state": "Fulfilled"
        }
        messages = [
            {
                "contentType": "PlainText",
                "content": "All right. Please feel free to talk with me whenever you feel bad."
            }
        ]
    response["sessionState"] = {
        "activeContexts": activeContexts,
        "sessionAttributes": sessionAttributes,
        "dialogAction": dialogAction,
        "intent": intent
    }
    response["messages"] = messages
    response["requestAttributes"] = {} if event.get("requestAttributes") is None else event["requestAttributes"]
    return response