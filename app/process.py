from .db import schemas
from typing import List

def get_summary(rows: List):
    summary = schemas.Summary().dict()

    for row in rows:
        summary["positive"] += row.positive
        summary["negative"] += row.negative
        summary["neutral"] += row.neutral
        summary["query"] += row.query
        summary["complain"] += row.complain
        summary["appreciation"] += row.appreciation
        summary["feedback"] += row.feedback
        summary["spam"] += row.spam
        summary["pi"] += row.pi
        summary["not_pi"] += row.not_pi
        summary["service_drop"] += row.service_drop

    summary["uid"] = row.uid
    summary["pgid"] = row.pgid
    return summary

def get_VisData(rows: List):
    tracker = {"Positive":{"Query":0,"Complain":0,"Appreciation":0,"Feedback":0,"Spam":0},
            "Negative":{"Query":0,"Complain":0,"Appreciation":0,"Feedback":0,"Spam":0},
            "Neutral":{"Query":0,"Complain":0,"Appreciation":0,"Feedback":0,"Spam":0},
            "All":{"Query":0,"Complain":0,"Appreciation":0,"Feedback":0,"Spam":0}}

    pos = 0
    neg = 0
    neut = 0

    for row in rows:
        if row.positive:
            pos += 1
            tracker["Positive"]["Query"] += row.query
            tracker["Positive"]["Complain"] += row.complain
            tracker["Positive"]["Appreciation"] += row.appreciation
            tracker["Positive"]["Feedback"] += row.feedback
            tracker["Positive"]["Spam"] += row.spam

            tracker["All"]["Query"] += row.query
            tracker["All"]["Complain"] += row.complain
            tracker["All"]["Appreciation"] += row.appreciation
            tracker["All"]["Feedback"] += row.feedback
            tracker["All"]["Spam"] += row.spam
        elif row.negative:
            neg += 1
            tracker["Negative"]["Query"] += row.query
            tracker["Negative"]["Complain"] += row.complain
            tracker["Negative"]["Appreciation"] += row.appreciation
            tracker["Negative"]["Feedback"] += row.feedback
            tracker["Negative"]["Spam"] += row.spam
            
            tracker["All"]["Query"] += row.query
            tracker["All"]["Complain"] += row.complain
            tracker["All"]["Appreciation"] += row.appreciation
            tracker["All"]["Feedback"] += row.feedback
            tracker["All"]["Spam"] += row.spam

        elif row.neutral:
            neut += 1
            tracker["Neutral"]["Query"] += row.query
            tracker["Neutral"]["Complain"] += row.complain
            tracker["Neutral"]["Appreciation"] += row.appreciation
            tracker["Neutral"]["Feedback"] += row.feedback
            tracker["Neutral"]["Spam"] += row.spam
            
            tracker["All"]["Query"] += row.query
            tracker["All"]["Complain"] += row.complain
            tracker["All"]["Appreciation"] += row.appreciation
            tracker["All"]["Feedback"] += row.feedback
            tracker["All"]["Spam"] += row.spam
            

    dataBar = []
    dataPie = [{"category": "Positive", "measure": pos/(pos+neg+neut)}, 
                {"category": "Negative", "measure": neg/(pos+neg+neut)}, 
                {"category": "Neutral", "measure": neut/(pos+neg+neut)}]

    for group in tracker.keys():
        for category in tracker[group].keys():
            dataBar.append({ "group": group, "category": category, "measure": tracker[group][category]})

    return {"BarChart":dataBar,"PieChart":dataPie}