"""
Bharat Banking - NLP Sentiment Service
Performs customer feedback sentiment analysis using HuggingFace Transformers.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("bharat_banking.nlp")

# Global pipeline instance
_sentiment_analyzer = None

def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from transformers import pipeline
            logger.info("Initializing HuggingFace sentiment pipeline...")
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            logger.warning(f"HuggingFace pipeline unavailable ({e}). Falling back to rule-based sentiment analyzer.")
            _sentiment_analyzer = None
    return _sentiment_analyzer


def rule_based_sentiment_fallback(clean_text: str) -> Dict[str, Any]:
    """
    Lightweight, reliable banking-domain sentiment evaluator used when
    deep learning transformer model is unavailable or encounters a network error.
    """
    lower = clean_text.lower()
    positive_keywords = [
        "swift", "seamless", "fast", "great", "excellent", "good", "satisfied", "satisfaction",
        "helpful", "courteous", "friendly", "polite", "quick", "best", "smooth", "efficient",
        "prompt", "supportive", "resolved", "recommend", "happy"
    ]
    negative_keywords = [
        "slow", "delay", "waited", "waiting", "queue", "rude", "poor", "worst", "terrible",
        "bad", "friction", "bottleneck", "crowded", "unhelpful", "horrible", "error", "issue",
        "complain", "complaint", "fail", "failed", "broken", "unresponsive", "disappointed"
    ]
    
    pos_score = sum(1 for w in positive_keywords if w in lower)
    neg_score = sum(1 for w in negative_keywords if w in lower)
    
    if pos_score > neg_score:
        label = "POSITIVE"
        conf = min(0.98, 0.70 + (pos_score * 0.08))
        expl = "Customer expresses satisfaction with branch service, staff responsiveness, or banking operations (evaluated via fallback NLP analyzer)."
    elif neg_score > pos_score:
        label = "NEGATIVE"
        conf = min(0.98, 0.70 + (neg_score * 0.08))
        expl = "Customer highlights operational friction, delay, or dissatisfaction requiring branch manager attention (evaluated via fallback NLP analyzer)."
    else:
        label = "NEUTRAL"
        conf = 0.60
        expl = "Neutral or balanced customer feedback containing both positive and critical elements."
        
    return {
        "status": "success",
        "label": label,
        "confidence": round(conf, 4),
        "explanation": expl
    }


def analyze_feedback(feedback_text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a given customer review string.
    Returns label, confidence score, and actionable operational takeaway.
    """
    clean_text = (feedback_text or "").strip()
    if not clean_text:
        return {
            "status": "error",
            "error": "Feedback text cannot be empty.",
            "label": "UNKNOWN",
            "confidence": 0.0,
            "explanation": "No text provided for analysis."
        }
    
    try:
        analyzer = get_sentiment_analyzer()
        if analyzer is not None:
            result = analyzer(clean_text[:512])[0]  # Cap length for transformer tokenizer
            label = result.get("label", "UNKNOWN").upper()
            confidence = round(float(result.get("score", 0.0)), 4)
            
            if label == "POSITIVE":
                explanation = "Customer expresses satisfaction with branch service, staff responsiveness, or banking operations."
            elif label == "NEGATIVE":
                explanation = "Customer highlights operational friction, delay, or dissatisfaction requiring branch manager attention."
            else:
                explanation = "Neutral or inconclusive customer sentiment."
                
            return {
                "status": "success",
                "label": label,
                "confidence": confidence,
                "explanation": explanation
            }
        else:
            return rule_based_sentiment_fallback(clean_text)
            
    except Exception as e:
        logger.warning(f"Error in primary sentiment transformer: {e}. Using rule-based fallback.")
        return rule_based_sentiment_fallback(clean_text)
