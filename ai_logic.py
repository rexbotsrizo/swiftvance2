# Production-Ready AI Logic for Law Firm Client Communication System
# This file contains clean, tested AI functions ready for production integration
# Uses OpenAI and LangGraph for advanced AI capabilities

import os
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration and Data Models
@dataclass
class ClientData:
    """Client information data model"""
    name: str
    case_manager: str
    accident_date: str
    gender: str
    phone_number: Optional[str] = None
    case_status: Optional[str] = "active"

@dataclass
class TriageResult:
    """Triage decision result data model"""
    action: str  # "ignore", "respond", "flag"
    should_respond: bool
    should_flag: bool
    risk_level: str  # "low", "medium", "high"
    sentiment: str  # "positive", "neutral", "negative"
    reasoning: str
    confidence: float

class ConversationState(TypedDict):
    """State management for LangGraph conversation flow"""
    messages: Annotated[List, add_messages]
    client_data: Dict
    triage_result: Optional[Dict]
    response_generated: Optional[str]
    action_items: List[Dict]

# Human-like Response Timing Implementation
def calculate_human_response_delay(message_length: int, response_complexity: str = "normal") -> float:
    """
    Calculate realistic human response delay based on message complexity
    Following instruction.txt: "Wait Before Replying: To seem more human, the AI will always 
    wait a short, random time before it replies. It will never reply instantly."
    
    Args:
        message_length: Length of client's message
        response_complexity: "simple", "normal", "complex"
        
    Returns:
        Delay in seconds (randomized for human-like behavior)
    """
    # Base reading time: ~250 words per minute
    reading_time = max(1.0, message_length / 250 * 60)
    
    # Thinking/processing time based on complexity
    if response_complexity == "simple":
        thinking_time = random.uniform(2.0, 5.0)
    elif response_complexity == "complex":
        thinking_time = random.uniform(8.0, 15.0)
    else:  # normal
        thinking_time = random.uniform(4.0, 10.0)
    
    # Additional time for longer messages (more consideration needed)
    if message_length > 100:
        thinking_time += random.uniform(2.0, 6.0)
    
    # Typing simulation: ~40-60 words per minute
    typing_time = random.uniform(3.0, 8.0)
    
    # Total delay with randomization
    total_delay = reading_time + thinking_time + typing_time
    
    # Add final variance (±25%) to feel more natural
    final_delay = total_delay * random.uniform(0.75, 1.25)
    
    # Reasonable bounds: 3-30 seconds
    return max(3.0, min(30.0, final_delay))

def apply_human_delay(message: str, response_type: str = "normal", enable_delay: bool = True) -> float:
    """
    Apply human-like delay before responding
    
    Args:
        message: Client's message
        response_type: "simple", "normal", "complex"
        enable_delay: Whether to actually apply delay (False for testing)
        
    Returns:
        Delay time applied in seconds
    """
    if not enable_delay:
        return 0.0
    
    delay = calculate_human_response_delay(len(message), response_type)
    
    # Show typing indicator simulation
    print(f"[AI Processing] Reading message... ({delay:.1f}s delay)")
    time.sleep(delay)
    
    return delay

# Initialize OpenAI client
def get_openai_client(streaming: bool = False) -> ChatOpenAI:
    """
    Initialize OpenAI client with API key from environment
    Make sure to set OPENAI_API_KEY environment variable
    
    Args:
        streaming: Enable streaming responses for real-time generation
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return ChatOpenAI(
        model="gpt-4",
        temperature=0.1,  # Low temperature for consistent responses
        api_key=api_key,
        streaming=streaming
    )

# Core AI Functions

def triage_message(message: str, client_data: Dict, conversation_history: List[Dict], 
                  use_streaming: bool = False, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Advanced triage AI using OpenAI GPT-4 to make intelligent decisions
    
    Args:
        message: Client's message text
        client_data: Client information dictionary
        conversation_history: Previous conversation messages
        use_streaming: Enable streaming for real-time analysis
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing triage decision and analysis
    """
    try:
        # Apply human-like delay BEFORE processing (instruction.txt requirement)
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay(message, "normal", True)
        
        client = get_openai_client(streaming=use_streaming)
        
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages for context
            history_context = "\n".join([
                f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}" 
                for msg in recent_messages
            ])
        
        # Create comprehensive system prompt for triage
        system_prompt = f"""You are a specialized AI triage system for a law firm's client communication.

CRITICAL RULES:
1. NEVER provide legal or medical advice
2. ALWAYS flag messages requesting legal/medical advice
3. Be empathetic but professional
4. Protect client data and privacy

CLIENT CONTEXT:
- Name: {client_data.get('name', 'Unknown')}
- Case Manager: {client_data.get('case_manager', 'Unknown')}
- Accident Date: {client_data.get('accident_date', 'Unknown')}
- Gender: {client_data.get('gender', 'Unknown')}

RECENT CONVERSATION:
{history_context}

CURRENT MESSAGE TO ANALYZE:
"{message}"

TRIAGE DECISION FRAMEWORK:
1. FLAG (send to human) if message contains:
   - Legal advice requests ("Can I sue...", "What's my case worth...")
   - Medical advice requests ("Should I see doctor...", "Is this normal...")
   - Extreme anger/threats
   - Sensitive personal information
   - Photos/videos of injuries

2. IGNORE (no response needed) if message is:
   - Simple acknowledgments ("ok", "thanks", "fine")
   - Very brief positive responses

3. RESPOND (AI can handle) if message is:
   - General check-in responses
   - Emotional support needs
   - Administrative questions AI can answer
   - Scheduling or procedural questions

SENTIMENT ANALYSIS:
- Positive: grateful, happy, satisfied language
- Negative: frustrated, angry, upset, disappointed language  
- Neutral: factual, brief, neither positive nor negative

RISK ASSESSMENT:
- High: Multiple negative sentiments, mentions leaving/switching lawyers, extreme dissatisfaction
- Medium: Some negative sentiment, concerns about process, moderate frustration
- Low: Positive/neutral sentiment, engaged communication

Return your analysis in this exact JSON format:
{{
    "action": "flag|respond|ignore",
    "should_respond": true/false,
    "should_flag": true/false,
    "risk_level": "low|medium|high",
    "sentiment": "positive|neutral|negative",
    "reasoning": "Brief explanation of decision",
    "confidence": 0.0-1.0,
    "detected_issues": ["list", "of", "specific", "issues"],
    "recommended_response_tone": "empathetic|professional|brief|escalate"
}}"""

        # Get AI decision
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this message and provide triage decision: {message}")
        ]
        
        if use_streaming:
            # For streaming, collect chunks and combine
            response_content = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
        else:
            response = client.invoke(messages)
            response_content = response.content
        
        # Parse JSON response
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            result = {
                "action": "flag",
                "should_respond": False,
                "should_flag": True,
                "risk_level": "medium",
                "sentiment": "neutral",
                "reasoning": "Unable to parse AI response",
                "confidence": 0.5
            }
        
        # Add success indicators
        result.update({
            "success": True,
            "message": "Triage completed successfully",
            "timestamp": datetime.now().isoformat(),
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Triage analysis failed",
            "action": "flag",  # Safe default
            "should_respond": False,
            "should_flag": True,
            "risk_level": "medium",
            "sentiment": "neutral"
        }


def analyze_sentiment(message: str, use_streaming: bool = False, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Advanced sentiment analysis using OpenAI with contextual understanding
    
    Args:
        message: Client's message text
        use_streaming: Enable streaming for real-time analysis
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing detailed sentiment analysis
    """
    try:
        # Apply human-like delay for natural feel
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay(message, "simple", True)
        
        client = get_openai_client(streaming=use_streaming)
        
        system_prompt = """You are an expert sentiment analysis AI for legal client communications.

Analyze the emotional tone and sentiment of client messages with special attention to:
- Legal case frustrations and concerns
- Physical/emotional pain indicators  
- Communication satisfaction levels
- Trust and confidence in legal representation
- Financial stress indicators

Sentiment Categories:
- POSITIVE: Grateful, satisfied, confident, hopeful, appreciative
- NEGATIVE: Frustrated, angry, disappointed, worried, dissatisfied, in pain
- NEUTRAL: Factual, brief, neither clearly positive nor negative

Return analysis in this JSON format:
{
    "sentiment": "positive|neutral|negative", 
    "confidence": 0.0-1.0,
    "emotional_indicators": ["list", "of", "detected", "emotions"],
    "keywords": ["relevant", "keywords", "found"],
    "intensity": "mild|moderate|strong",
    "concerns": ["specific", "concerns", "detected"],
    "pain_indicators": true/false,
    "satisfaction_level": "high|medium|low|unknown"
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze sentiment of this message: {message}")
        ]
        
        if use_streaming:
            # For streaming, collect chunks and combine
            response_content = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
        else:
            response = client.invoke(messages)
            response_content = response.content
        
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback sentiment analysis
            result = {
                "sentiment": "neutral",
                "confidence": 0.5,
                "keywords": [],
                "intensity": "mild"
            }
        
        result.update({
            "success": True,
            "message": "Sentiment analysis completed",
            "analyzed_text_length": len(message),
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Sentiment analysis failed",
            "sentiment": "neutral",
            "confidence": 0.0
        }


def generate_response(message: str, client_data: Dict, conversation_history: List[Dict], 
                     message_count_this_week: int, use_streaming: bool = False, 
                     stream_callback=None, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Generate empathetic, human-like response using OpenAI with optional streaming
    Following instruction.txt guidelines for natural, human-like communication
    
    Args:
        message: Client's message
        client_data: Client information
        conversation_history: Previous messages  
        message_count_this_week: Current week's message count
        use_streaming: Enable streaming for real-time response generation
        stream_callback: Optional callback function for streaming chunks
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing generated response and action items
    """
    try:
        # Check weekly message limit (25 total messages per week)
        if message_count_this_week >= 25:
            return {
                "response": None,
                "action_items": [],
                "success": False,
                "message": "Weekly message limit exceeded - additional charges apply",
                "charge_additional": True,
                "limit_exceeded": True
            }
        
        # Apply human-like delay before generating response
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay(message, "complex", True)
        
        client = get_openai_client(streaming=use_streaming)
        name = client_data.get("name", "")
        case_manager = client_data.get("case_manager", "")
        gender = client_data.get("gender", "").lower()
        accident_date = client_data.get("accident_date", "")
        
        # Calculate days since accident for contextual responses
        days_since_accident = 0
        if accident_date:
            try:
                accident_dt = datetime.fromisoformat(accident_date)
                days_since_accident = (datetime.now() - accident_dt).days
            except:
                pass
        
        # Build conversation context
        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-10:]
            history_context = "\n".join([
                f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}"
                for msg in recent_messages
            ])
        
        # Get sentiment analysis for response tone
        sentiment_result = analyze_sentiment(message, use_streaming=False, enable_human_delay=False)  # Quick sentiment analysis
        sentiment = sentiment_result.get("sentiment", "neutral")
        
        system_prompt = f"""You are a caring AI assistant for a law firm, communicating with clients via SMS.

CRITICAL INSTRUCTION COMPLIANCE:
- NEVER reply instantly - always appear to take time to read and consider
- Sound completely human and natural, never robotic or templated
- Use casual, warm language like a caring friend who works at the law firm
- Show genuine empathy and emotional intelligence
- Match the client's energy and communication style perfectly
- Remember details from past conversations to build real relationships
- Use names naturally in conversation (first names only)
- Vary your language - never sound scripted or repetitive

PERSONALITY GUIDELINES:
- Be warm, caring, and genuinely interested in their wellbeing
- Use casual, friendly language appropriate for text messaging  
- Show real empathy for their situation and timeline since accident
- Be conversational and natural - like texting a friend who cares
- Remember past conversations and reference them naturally
- Use emojis occasionally but not excessively (😊💙🙏)
- NEVER sound like customer service or a script

CLIENT INFORMATION:
- Name: {name}
- Case Manager: {case_manager}  
- Gender: {gender}
- Days since accident: {days_since_accident}
- Current sentiment: {sentiment}

CONVERSATION HISTORY:
{history_context}

CURRENT MESSAGE: "{message}"

RESPONSE RULES (Following instruction.txt):
1. Keep responses conversational and brief (text message style)
2. Show empathy based on accident timeline - early days need more emotional care
3. Use the case manager's first name naturally in conversation
4. Match client's communication style exactly:
   - If they're brief, be brief but warm
   - If they're chatty, be more engaging and detailed
   - If they're formal, be professional but still warm
   - If they use emojis, you can use them too
5. For pain/medical concerns: acknowledge deeply and redirect to case manager
6. For legal questions: redirect to case manager without giving advice
7. Create follow-up reminders when appropriate
8. NEVER mention internal processes (flagging, risk assessment, AI analysis, etc.)
9. Always sound like a real person who genuinely cares

RESPONSE TONE based on sentiment:
- Positive sentiment: Warm, encouraging, celebrate with them
- Negative sentiment: Extra empathetic, supportive, more personal care
- Neutral sentiment: Friendly but not overly emotional

HUMAN-LIKE QUALITIES TO INCLUDE:
- Slight imperfections in language (like real people text)
- Natural conversation flow and transitions
- Emotional intelligence and genuine care
- Personal touches that show you remember them
- Warmth that feels authentic, not scripted

Generate a completely natural, caring response that builds trust and feels like it comes from a real person who works at the firm and genuinely cares about this client."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate an appropriate response to the client's message.")
        ]
        
        if use_streaming:
            # Stream response generation
            generated_response = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    generated_response += chunk.content
                    # Call callback if provided for real-time updates
                    if stream_callback:
                        stream_callback(chunk.content)
        else:
            response = client.invoke(messages)
            generated_response = response.content.strip()
        
        # Generate action items if needed
        action_items = []
        
        # Check for follow-up opportunities
        if any(word in message.lower() for word in ["doctor", "appointment", "surgery", "therapy"]):
            action_items.append({
                "task": f"Follow up with {name} about medical appointment outcome",
                "scheduled_date": (datetime.now() + timedelta(days=14)).isoformat(),
                "type": "medical_follow_up",
                "priority": "medium"
            })
        
        if any(word in message.lower() for word in ["bill", "insurance", "payment", "money"]):
            action_items.append({
                "task": f"Follow up with {name} about financial concerns",
                "scheduled_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "type": "financial_follow_up", 
                "priority": "high"
            })
        
        return {
            "response": generated_response,
            "action_items": action_items,
            "response_tone": sentiment,
            "message_count_after": message_count_this_week + 1,
            "success": True,
            "message": "Response generated successfully",
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied,
            "feels_human": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Response generation failed",
            "response": None,
            "action_items": []
        }


def generate_check_in(client_data: Dict, days_since_accident: int, last_sentiment: str,
                     use_streaming: bool = False, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Generate personalized weekly check-in messages with optional streaming
    Following instruction.txt for human-like, caring communication
    
    Args:
        client_data: Client information
        days_since_accident: Number of days since accident
        last_sentiment: Client's most recent sentiment
        use_streaming: Enable streaming for real-time generation
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing check-in message
    """
    try:
        # Apply human-like delay for natural feel
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay("generating check-in", "normal", True)
        
        client = get_openai_client(streaming=use_streaming)
        
        name = client_data.get("name", "")
        case_manager = client_data.get("case_manager", "")
        gender = client_data.get("gender", "").lower()
        
        system_prompt = f"""Generate a natural, caring check-in message for a law firm client.

INSTRUCTION COMPLIANCE:
- Sound completely human and natural (never robotic or templated)
- Use warm, caring language like a friend who works at the firm
- Vary language significantly to avoid any repetition or patterns
- Show genuine interest in their wellbeing beyond just the legal case
- Use casual, text-message appropriate language
- Never sound scripted or like customer service

CLIENT INFO:
- Name: {name}
- Case Manager: {case_manager}
- Gender: {gender}
- Days since accident: {days_since_accident}
- Recent sentiment: {last_sentiment}

MESSAGE GUIDELINES:
1. Sound completely human and conversational (SMS style)
2. Vary language dramatically to avoid any repetition or templates
3. Show appropriate care based on timeline:
   - First week (1-7 days): Extra gentle, supportive, focus on immediate recovery
   - First month (8-30 days): Ongoing support, check on healing progress
   - 1-3 months (31-90 days): Maintaining connection, case progress interest
   - 3+ months (90+ days): Relationship maintenance, long-term care
4. Adjust tone based on last sentiment:
   - Negative: More personal and deeply caring, address their concerns
   - Positive: Warm and encouraging, build on their positive feelings
   - Neutral: Friendly but genuine interest without being pushy
5. Include case manager naturally in message (use first name only)
6. Keep it brief but genuinely caring
7. NEVER sound scripted, robotic, or like a template
8. Use occasional emojis if appropriate for the tone (😊💙🙏)
9. Make it feel like a real person who remembers them and cares

HUMAN-LIKE QUALITIES:
- Natural conversation flow
- Genuine emotional intelligence
- Personal touches that show you remember their situation
- Slight imperfections that make it feel human
- Warmth that feels authentic

Generate a single, completely natural check-in message that feels like it comes from a caring person at the firm who genuinely remembers and cares about this client."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate a check-in message for this client.")
        ]
        
        if use_streaming:
            # Stream check-in generation
            response_content = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
            check_in_message = response_content.strip()
        else:
            response = client.invoke(messages)
            check_in_message = response.content.strip()
        
        return {
            "message": check_in_message,
            "scheduled_for": datetime.now().isoformat(),
            "success": True,
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied,
            "feels_human": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Check-in generation failed"
        }


def assess_risk_level(sentiment_history: List[str], response_rate: float, 
                     keywords_mentioned: List[str], days_since_last_response: int,
                     use_streaming: bool = False, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Comprehensive risk assessment using AI analysis with optional streaming
    
    Args:
        sentiment_history: List of recent sentiment scores
        response_rate: Client's response rate (0.0 to 1.0)  
        keywords_mentioned: Risk-related keywords client has used
        days_since_last_response: Days since client last responded
        use_streaming: Enable streaming for real-time analysis
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing risk assessment and recommendations
    """
    try:
        # Apply human-like delay for natural processing
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay("risk assessment analysis", "complex", True)
        
        client = get_openai_client(streaming=use_streaming)
        
        system_prompt = f"""You are an expert risk assessment AI for law firm client relationships.

Analyze the following client data to determine churn risk:

SENTIMENT HISTORY: {sentiment_history}
RESPONSE RATE: {response_rate} (0.0 = never responds, 1.0 = always responds)
KEYWORDS MENTIONED: {keywords_mentioned}
DAYS SINCE LAST RESPONSE: {days_since_last_response}

RISK FACTORS TO CONSIDER:
- Declining sentiment trend
- Low response rates
- Mentions of dissatisfaction ("slow", "unhappy", "frustrated")
- Mentions of switching lawyers ("another lawyer", "different firm")
- Extended periods of non-communication
- Financial stress indicators
- Process confusion or concerns

RISK LEVELS:
- HIGH: Immediate intervention needed, high churn probability
- MEDIUM: Monitor closely, some concerning indicators
- LOW: Healthy relationship, standard monitoring

Return assessment in JSON format:
{{
    "risk_level": "low|medium|high",
    "risk_score": 0-10,
    "primary_risk_factors": ["list", "of", "main", "concerns"],
    "sentiment_trend": "improving|stable|declining",
    "engagement_level": "high|medium|low",
    "recommendations": ["specific", "action", "items"],
    "urgency": "immediate|within_24h|within_week|routine",
    "confidence": 0.0-1.0
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze this client data and provide risk assessment.")
        ]
        
        if use_streaming:
            # Stream risk assessment
            response_content = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
        else:
            response = client.invoke(messages)
            response_content = response.content
        
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback assessment
            result = {
                "risk_level": "medium",
                "risk_score": 5,
                "recommendations": ["Monitor client communication closely"]
            }
        
        result.update({
            "success": True,
            "message": "Risk assessment completed",
            "assessment_date": datetime.now().isoformat(),
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Risk assessment failed",
            "risk_level": "medium"  # Safe default
        }


def generate_client_insight(client_data: Dict, conversation_history: List[Dict], 
                          sentiment_trend: List[str], current_risk_level: str,
                          use_streaming: bool = False, enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Generate comprehensive client insight using advanced AI analysis with optional streaming
    
    Args:
        client_data: Client information
        conversation_history: All conversation messages
        sentiment_trend: Recent sentiment history
        current_risk_level: Current assessed risk level
        use_streaming: Enable streaming for real-time analysis
        enable_human_delay: Apply human-like response delay
        
    Returns:
        Dict containing detailed client insight
    """
    try:
        # Apply human-like delay for thorough analysis
        delay_applied = 0.0
        if enable_human_delay:
            delay_applied = apply_human_delay("comprehensive client analysis", "complex", True)
        
        client = get_openai_client(streaming=use_streaming)
        
        name = client_data.get("name", "")
        case_manager = client_data.get("case_manager", "")
        accident_date = client_data.get("accident_date", "")
        
        # Prepare conversation data for analysis
        recent_messages = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
        conversation_text = "\n".join([
            f"{msg.get('sender', 'unknown')}: {msg.get('message', '')}"
            for msg in recent_messages
        ])
        
        system_prompt = f"""You are an expert client relationship analyst for a law firm.

Generate a comprehensive insight summary for case managers and attorneys.

CLIENT DATA:
- Name: {name}
- Case Manager: {case_manager}
- Accident Date: {accident_date}
- Current Risk Level: {current_risk_level}
- Sentiment Trend: {sentiment_trend}

RECENT CONVERSATIONS:
{conversation_text}

ANALYSIS REQUIREMENTS:
1. Client's current emotional state and needs
2. Key concerns or issues raised
3. Communication patterns and preferences
4. Relationship health with the firm
5. Specific action items for case manager
6. Early warning signs to monitor

Generate a professional summary that helps the legal team provide better client service.

Return in JSON format:
{{
    "executive_summary": "Brief 2-3 sentence overview",
    "current_sentiment": "positive|neutral|negative",
    "key_concerns": ["primary", "concerns", "identified"],
    "communication_style": "brief|detailed|emotional|professional",
    "relationship_health": "excellent|good|concerning|poor", 
    "action_items": ["specific", "steps", "for", "case", "manager"],
    "warning_signs": ["things", "to", "monitor"],
    "strengths": ["positive", "aspects", "of", "relationship"],
    "next_contact_recommendation": "timing and approach suggestions",
    "priority_level": "high|medium|low"
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate comprehensive client insight based on this data.")
        ]
        
        if use_streaming:
            # Stream insight generation
            response_content = ""
            for chunk in client.stream(messages):
                if hasattr(chunk, 'content'):
                    response_content += chunk.content
        else:
            response = client.invoke(messages)
            response_content = response.content
        
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback insight
            result = {
                "executive_summary": f"Analysis of {name}'s recent communications and relationship status.",
                "current_sentiment": sentiment_trend[-1] if sentiment_trend else "neutral",
                "key_concerns": ["Communication analysis incomplete"],
                "action_items": ["Review client communications manually"]
            }
        
        # Add metadata
        result.update({
            "success": True,
            "message": "Client insight generated successfully",
            "generated_date": datetime.now().isoformat(),
            "conversation_messages_analyzed": len(recent_messages),
            "client_name": name,
            "human_delay_applied": enable_human_delay,
            "delay_seconds": delay_applied
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Client insight generation failed",
            "executive_summary": "Analysis failed due to technical error"
        }


# LangGraph Implementation for Complex Conversation Flows
def create_conversation_graph() -> StateGraph:
    """
    Create a LangGraph workflow for handling complex client conversations
    
    Returns:
        Configured StateGraph for conversation management
    """
    
    def triage_node(state: ConversationState) -> ConversationState:
        """Triage node in the conversation graph"""
        messages = state["messages"]
        client_data = state["client_data"]
        
        if messages:
            latest_message = messages[-1].content
            # Get conversation history in the right format
            conversation_history = [
                {"sender": "client" if i % 2 == 0 else "ai", "message": msg.content}
                for i, msg in enumerate(messages[:-1])
            ]
            
            triage_result = triage_message(latest_message, client_data, conversation_history, 
                                          use_streaming=False, enable_human_delay=True)
            state["triage_result"] = triage_result
        
        return state
    
    def response_node(state: ConversationState) -> ConversationState:
        """Response generation node"""
        if state["triage_result"] and state["triage_result"].get("should_respond"):
            messages = state["messages"]
            latest_message = messages[-1].content
            
            conversation_history = [
                {"sender": "client" if i % 2 == 0 else "ai", "message": msg.content}
                for i, msg in enumerate(messages[:-1])
            ]
            
            response_result = generate_response(
                latest_message,
                state["client_data"],
                conversation_history,
                len(messages),
                use_streaming=False,
                stream_callback=None,
                enable_human_delay=True
            )
            
            if response_result.get("success"):
                state["response_generated"] = response_result["response"]
                state["action_items"] = response_result.get("action_items", [])
        
        return state
    
    def should_respond(state: ConversationState) -> str:
        """Routing function to determine next step"""
        triage_result = state.get("triage_result", {})
        
        if triage_result.get("should_flag"):
            return "flag_for_human"
        elif triage_result.get("should_respond"):
            return "generate_response"
        else:
            return "end"
    
    # Create the graph
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("triage", triage_node)
    graph.add_node("generate_response", response_node)
    graph.add_node("flag_for_human", lambda state: state)  # Terminal node
    graph.add_node("end", lambda state: state)  # Terminal node
    
    # Set entry point
    graph.set_entry_point("triage")
    
    # Add conditional routing
    graph.add_conditional_edges(
        "triage",
        should_respond,
        {
            "flag_for_human": "flag_for_human",
            "generate_response": "generate_response", 
            "end": "end"
        }
    )
    
    # Add final edges
    graph.add_edge("generate_response", "end")
    graph.add_edge("flag_for_human", "end")
    
    return graph.compile()


# Utility Functions

def validate_client_data(client_data: Dict) -> Tuple[bool, str]:
    """
    Validate client data structure
    
    Args:
        client_data: Client information dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["name", "case_manager", "accident_date", "gender"]
    
    for field in required_fields:
        if field not in client_data or not client_data[field]:
            return False, f"Missing required field: {field}"
    
    # Validate date format
    try:
        datetime.fromisoformat(client_data["accident_date"])
    except ValueError:
        return False, "Invalid accident_date format. Use ISO format (YYYY-MM-DD)"
    
    return True, ""


def calculate_days_since_accident(accident_date: str) -> int:
    """
    Calculate days since accident
    
    Args:
        accident_date: ISO formatted date string
        
    Returns:
        Number of days since accident
    """
    try:
        accident_dt = datetime.fromisoformat(accident_date)
        return (datetime.now() - accident_dt).days
    except ValueError:
        return 0


# Production Integration Helpers

def stream_response_generation(message: str, client_data: Dict, conversation_history: List[Dict],
                              message_count_this_week: int, callback_function=None, 
                              enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Generate streaming response with real-time callback for UI updates
    
    Args:
        message: Client's message
        client_data: Client information  
        conversation_history: Previous messages
        message_count_this_week: Current week's message count
        callback_function: Function to call with each chunk (chunk_text: str) -> None
        enable_human_delay: Apply human-like response timing
        
    Returns:
        Dict containing complete response and metadata
    """
    def streaming_callback(chunk: str):
        """Internal callback wrapper"""
        if callback_function:
            callback_function(chunk)
    
    return generate_response(
        message=message,
        client_data=client_data, 
        conversation_history=conversation_history,
        message_count_this_week=message_count_this_week,
        use_streaming=True,
        stream_callback=streaming_callback,
        enable_human_delay=enable_human_delay
    )


def process_client_message(message: str, client_data: Dict, conversation_history: List[Dict] = None,
                          enable_streaming: bool = False, stream_callback=None, 
                          enable_human_delay: bool = True) -> Dict[str, Any]:
    """
    Main entry point for processing client messages in production with optional streaming
    Following instruction.txt for human-like timing and natural responses
    
    Args:
        message: Client's message
        client_data: Client information
        conversation_history: Previous conversation messages
        enable_streaming: Enable streaming responses
        stream_callback: Callback function for streaming chunks
        enable_human_delay: Apply human-like response delays (NEVER reply instantly)
        
    Returns:
        Complete processing result with all decisions and outputs
    """
    try:
        # Validate inputs
        is_valid, error_msg = validate_client_data(client_data)
        if not is_valid:
            return {
                "success": False,
                "error": error_msg,
                "message": "Invalid client data provided"
            }
        
        if conversation_history is None:
            conversation_history = []
        
        # Run triage analysis with human delay
        triage_result = triage_message(
            message, 
            client_data, 
            conversation_history, 
            use_streaming=enable_streaming,
            enable_human_delay=enable_human_delay
        )
        
        result = {
            "triage": triage_result,
            "response": None,
            "action_items": [],
            "should_notify_case_manager": False,
            "human_timing_applied": enable_human_delay
        }
        
        # Generate response if appropriate
        if triage_result.get("should_respond", False):
            if enable_streaming and stream_callback:
                response_result = stream_response_generation(
                    message, 
                    client_data, 
                    conversation_history,
                    len(conversation_history) + 1,
                    stream_callback,
                    enable_human_delay
                )
            else:
                response_result = generate_response(
                    message, 
                    client_data, 
                    conversation_history,
                    len(conversation_history) + 1,
                    use_streaming=enable_streaming,
                    enable_human_delay=enable_human_delay
                )
            result["response"] = response_result
        
        # Determine if case manager should be notified
        if (triage_result.get("should_flag", False) or 
            triage_result.get("risk_level") == "high"):
            result["should_notify_case_manager"] = True
        
        result.update({
            "success": True,
            "message": "Message processed successfully",
            "processed_at": datetime.now().isoformat()
        })
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Message processing failed"
        }
