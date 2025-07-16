"""
Arviso AI Agent - Client Message Processing System
Uses LangGraph and OpenAI reasoning models for intelligent client communication handling
"""

import json
import time
import random
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientMessageState(TypedDict):
    """State for the client message processing graph"""
    message: str
    client_details: Dict[str, Any]
    message_count: int
    enable_human_delay: bool
    conversation_history: List[Dict]
    
    # Processing state
    sentiment_analysis: Optional[Dict]
    concern_level: Optional[str]
    flag_decision: Optional[Dict]
    response_decision: Optional[Dict]
    response_content: Optional[str]
    final_result: Optional[Dict]
    messages: List[Dict]

class ArvisoAIAgent:
    """Main AI Agent for processing client messages using LangGraph and OpenAI reasoning"""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the AI agent with OpenAI reasoning model"""
        # Use provided API key or get from environment
        api_key = os.getenv("OPENAI_API_KEY")
        

        try:
            self.reasoning_model = ChatOpenAI(
                model="gpt-4-1106-preview",  # Best for structured reasoning
                temperature=0.1,
                api_key=api_key
            )
            
            self.chat_model = ChatOpenAI(
                model="gpt-4-0125-preview",  # Best for rich, high-context conversations
                temperature=0.45,
                api_key=api_key
            )
            
        
        self.graph = self._build_processing_graph()
    
    def _build_processing_graph(self):
        """Build the LangGraph for processing client messages"""
        
        # Create state graph
        workflow = StateGraph(ClientMessageState)
        
        # Add nodes
        workflow.add_node("analyze_sentiment", self._analyze_sentiment)
        workflow.add_node("assess_concern_level", self._assess_concern_level)
        workflow.add_node("decide_flag", self._decide_flag)
        workflow.add_node("decide_response", self._decide_response)
        workflow.add_node("generate_response_content", self._generate_response_content)
        workflow.add_node("compile_results", self._compile_results)
        
        # Add edges
        workflow.add_edge(START, "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "assess_concern_level")
        workflow.add_edge("assess_concern_level", "decide_flag")
        workflow.add_edge("decide_flag", "decide_response")
        workflow.add_edge("decide_response", "generate_response_content")
        workflow.add_edge("generate_response_content", "compile_results")
        workflow.add_edge("compile_results", END)
        
        return workflow.compile()
    
    def _analyze_sentiment(self, state: ClientMessageState) -> ClientMessageState:
        """Analyze sentiment using OpenAI reasoning model with enhanced context"""
        
        client_context = self._build_client_context(state["client_details"])
        conversation_context = self._build_conversation_context(state["conversation_history"])
        
        # Extract conversation insights for sentiment analysis
        conversation_insights = ""
        if state["conversation_history"]:
            total_messages = len(state["conversation_history"])
            client_messages = [msg for msg in state["conversation_history"] if msg.get("sender") == "client"]
            
            conversation_insights = f"""
            CONVERSATION INSIGHTS:
            - Total conversation messages: {total_messages}
            - Client messages: {len(client_messages)}
            - Current message number: {state['message_count']}
            - This provides context for communication patterns and client engagement level
            """
        
        sentiment_prompt = f"""
        You are an expert sentiment analysis agent for a law firm's client communication system.
        
        CLIENT CONTEXT:
        {client_context}
        
        CONVERSATION HISTORY AND PATTERNS:
        {conversation_context}
        
        {conversation_insights}
        
        CURRENT MESSAGE TO ANALYZE:
        "{state['message']}"
        
        ANALYSIS REQUIREMENTS:
        1. Analyze the sentiment of the current message (positive, neutral, negative)
        2. Consider the client's journey stage and relationship timeline
        3. Look for emotional indicators specific to legal and medical case stress
        4. Identify key topics and concerns mentioned
        5. Assess sentiment trend based on conversation patterns and history
        6. Consider if this message represents a change from previous communication patterns
        
        IMPORTANT: Use the conversation history to understand context - if client was previously positive but now negative, this is significant. If they've been asking the same questions repeatedly, this shows frustration building.
        
        Provide your analysis in JSON format:
        {{
            "sentiment": "positive|neutral|negative",
            "confidence": 0.0-1.0,
            "emotional_indicators": ["list", "of", "indicators"],
            "key_topics": ["list", "of", "topics"],
            "sentiment_trend": "improving|stable|declining",
            "pattern_change": "escalating|consistent|de-escalating",
            "reasoning": "Your detailed reasoning considering conversation history and patterns"
        }}
        """
        
        try:
            response = self.reasoning_model.invoke([
                SystemMessage(content="You are an expert sentiment analysis agent. Provide thorough, reasoned analysis."),
                HumanMessage(content=sentiment_prompt)
            ])
            
            sentiment_result = self._parse_json_response(response.content)
            
            # Ensure sentiment_result is a dictionary with proper defaults
            if not isinstance(sentiment_result, dict):
                logger.warning(f"Sentiment result is not a dict: {type(sentiment_result)}, using defaults")
                sentiment_result = {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "emotional_indicators": [],
                    "key_topics": [],
                    "sentiment_trend": "stable",
                    "pattern_change": "consistent",
                    "reasoning": "Failed to parse sentiment analysis"
                }
            
            # Validate sentiment value
            sentiment = sentiment_result.get("sentiment", "neutral")
            if sentiment not in ["positive", "neutral", "negative"]:
                logger.warning(f"Invalid sentiment: {sentiment}, defaulting to neutral")
                sentiment_result["sentiment"] = "neutral"
            
            state["sentiment_analysis"] = sentiment_result
            
            logger.info(f"Sentiment analysis completed: {sentiment_result.get('sentiment', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            state["sentiment_analysis"] = {
                "sentiment": "neutral",
                "confidence": 0.5,
                "emotional_indicators": [],
                "key_topics": [],
                "sentiment_trend": "stable",
                "reasoning": "Analysis failed, defaulting to neutral"
            }
        
        return state
    
    def _assess_concern_level(self, state: ClientMessageState) -> ClientMessageState:
        """Assess concern level using OpenAI reasoning model with conversation awareness"""
        
        client_context = self._build_client_context(state["client_details"])
        conversation_context = self._build_conversation_context(state["conversation_history"])
        sentiment_data = state["sentiment_analysis"]
        
        # Analyze conversation frequency and patterns
        conversation_insights = ""
        if state["conversation_history"]:
            client_messages = [msg for msg in state["conversation_history"] if msg.get("sender") == "client"]
            recent_client_messages = client_messages[-10:]  # Last 10 client messages
            
            # Check for repeated questions or concerns
            repeated_concerns = []
            for i, msg in enumerate(recent_client_messages):
                content = msg.get("content", "").lower()
                for j, other_msg in enumerate(recent_client_messages):
                    if i != j and other_msg.get("content", "").lower() in content:
                        repeated_concerns.append("Similar questions asked multiple times")
                        break
            
            conversation_insights = f"""
            CONVERSATION PATTERN ANALYSIS:
            - Recent client messages: {len(recent_client_messages)}
            - Message frequency suggests: {'High engagement' if len(recent_client_messages) > 5 else 'Normal engagement'}
            - Repeated concerns detected: {len(repeated_concerns) > 0}
            - Pattern change from history: {sentiment_data.get('pattern_change', 'unknown')}
            """
        
        concern_prompt = f"""
        You are an expert risk assessment agent for a law firm's client retention system.
        
        CLIENT CONTEXT:
        {client_context}
        
        CONVERSATION HISTORY AND PATTERNS:
        {conversation_context}
        
        {conversation_insights}
        
        CURRENT MESSAGE:
        "{state['message']}"
        
        SENTIMENT ANALYSIS:
        {json.dumps(sentiment_data, indent=2)}
        
        MESSAGE COUNT IN CONVERSATION: {state['message_count']}
        
        CONCERN LEVEL ASSESSMENT CRITERIA:
        
        HIGH CONCERN INDICATORS:
        - Mentions of other law firms or attorneys
        - Threats to leave or fire the firm
        - Extreme anger or frustration
        - Mentions of feeling "forgotten" or "ignored"
        - Financial distress affecting case decisions
        - Unrealistic timeline expectations with anger
        - Repeated questions without satisfactory answers (from conversation history)
        - Pattern change from positive to negative communication
        - Escalating tone compared to previous messages
        
        MEDIUM CONCERN INDICATORS:
        - Mild frustration about case progress
        - Questions about timeline repeatedly
        - Concerns about communication frequency
        - Mild dissatisfaction with service
        - Anxiety about case outcome
        - Increased message frequency indicating stress
        
        LOW CONCERN INDICATORS:
        - Routine questions about case
        - Positive or neutral sentiment
        - Normal check-ins and responses
        - Gratitude or satisfaction expressed
        - Consistent communication patterns
        
        IMPORTANT: Use conversation history to identify patterns - if client previously expressed satisfaction but now shows concern, this is significant. Repeated questions suggest unmet needs.
        
        Assess the concern level and provide reasoning in JSON format:
        {{
            "concern_level": "low|medium|high",
            "risk_indicators": ["list", "of", "specific", "indicators"],
            "client_retention_risk": 0.0-1.0,
            "historical_context": "How this message compares to previous communication patterns",
            "reasoning": "Your detailed reasoning considering conversation history and patterns",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = self.reasoning_model.invoke([
                SystemMessage(content="You are an expert risk assessment agent. Provide thorough analysis for client retention."),
                HumanMessage(content=concern_prompt)
            ])
            
            concern_result = self._parse_json_response(response.content)
            
            # Ensure concern_result is a dictionary with proper defaults
            if not isinstance(concern_result, dict):
                logger.warning(f"Concern result is not a dict: {type(concern_result)}, using defaults")
                concern_result = {
                    "concern_level": "medium",
                    "reasoning": "Failed to parse concern assessment",
                    "confidence": 0.5
                }
            
            # Ensure we have a valid concern level
            concern_level = concern_result.get("concern_level", "medium")
            if concern_level not in ["low", "medium", "high", "critical"]:
                logger.warning(f"Invalid concern level: {concern_level}, defaulting to medium")
                concern_level = "medium"
            
            state["concern_level"] = concern_level
            
            logger.info(f"Concern level assessed: {concern_result.get('concern_level', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Concern level assessment failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            state["concern_level"] = "medium"
        
        return state
    
    def _decide_flag(self, state: ClientMessageState) -> ClientMessageState:
        """Decide whether to flag message using OpenAI reasoning model with conversation context"""
        
        client_context = self._build_client_context(state["client_details"])
        conversation_context = self._build_conversation_context(state["conversation_history"])
        sentiment_data = state["sentiment_analysis"]
        
        # Analyze if similar issues were flagged before
        conversation_insights = ""
        if state["conversation_history"]:
            system_responses = [msg for msg in state["conversation_history"] if msg.get("sender") == "system"]
            flagged_indicators = ["forwarded", "escalated", "manager", "attorney", "review"]
            
            recent_flags = 0
            for msg in system_responses[-5:]:
                content_lower = msg.get("content", "").lower()
                if any(word in content_lower for word in flagged_indicators):
                    recent_flags += 1
            
            conversation_insights = f"""
            FLAGGING HISTORY CONTEXT:
            - Recent system responses with flag indicators: {recent_flags}
            - Pattern suggests: {'Escalation pattern' if recent_flags > 1 else 'Normal handling'}
            - Client communication pattern: {sentiment_data.get('pattern_change', 'consistent')}
            """
        
        flag_prompt = f"""
        You are an expert triage agent for a law firm's AI communication system.
        
        CLIENT CONTEXT:
        {client_context}
        
        CONVERSATION HISTORY AND PATTERNS:
        {conversation_context}
        
        {conversation_insights}
        
        CURRENT MESSAGE:
        "{state['message']}"
        
        SENTIMENT ANALYSIS:
        {json.dumps(sentiment_data, indent=2)}
        
        CONCERN LEVEL: {state['concern_level']}
        
        FLAGGING CRITERIA (MUST FLAG):
        
        1. LEGAL/MEDICAL ADVICE REQUESTS:
        - Questions about legal strategy or case value
        - Requests for medical advice or treatment decisions
        - Questions about settlement amounts or case worth
        - Legal procedure questions beyond basic information
        
        2. EXTREME EMOTIONAL DISTRESS:
        - Threats of self-harm or extreme despair
        - Extreme anger with threats or abusive language
        - Mentions of wanting to "give up" on case
        
        3. SENSITIVE CONTENT:
        - Photos/videos of injuries or accident scenes
        - Medical records or sensitive documents
        - Insurance correspondence or legal documents
        
        4. ESCALATION INDICATORS:
        - Mentions of other attorneys or law firms
        - Threats to fire the firm or leave
        - Complaints about specific staff members
        - Demands for immediate attorney contact
        
        5. URGENT CASE MATTERS:
        - New injuries or medical complications
        - Insurance company contact or pressure
        - Court dates or legal deadlines mentioned
        - Accident-related developments
        
        6. CONVERSATION PATTERN CONCERNS:
        - Repeated questions not being answered adequately
        - Escalating frustration pattern from conversation history
        - Significant sentiment change from positive to negative
        
        DECISION RULES:
        - If ANY flagging criteria is met, MUST flag
        - High concern level alone may warrant flagging
        - Consider conversation history - if client was recently satisfied but now distressed, flag
        - Pattern changes and repeated concerns should influence decision
        - When in doubt, err on the side of flagging
        - AI should NEVER handle sensitive legal/medical matters
        
        IMPORTANT: Use conversation history to identify if this is part of an escalating pattern or a new concern that requires human attention.
        
        Provide your flagging decision in JSON format:
        {{
            "should_flag": true|false,
            "flag_reasons": ["list", "of", "specific", "reasons"],
            "urgency_level": "low|medium|high|critical",
            "pattern_context": "How conversation history influenced this decision",
            "reasoning": "Your detailed reasoning considering conversation patterns and history",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = self.reasoning_model.invoke([
                SystemMessage(content="You are an expert triage agent. Prioritize client safety and legal compliance."),
                HumanMessage(content=flag_prompt)
            ])
            
            flag_result = self._parse_json_response(response.content)
            
            # Ensure flag_result is a dictionary with proper defaults
            if not isinstance(flag_result, dict):
                logger.warning(f"Flag result is not a dict: {type(flag_result)}, using defaults")
                flag_result = {
                    "should_flag": False,
                    "flag_reasons": [],
                    "urgency_level": "low",
                    "reasoning": "Failed to parse flag decision",
                    "confidence": 0.5
                }
            
            # Ensure boolean value for should_flag
            should_flag = flag_result.get("should_flag", False)
            if not isinstance(should_flag, bool):
                logger.warning(f"should_flag is not boolean: {type(should_flag)}, converting")
                should_flag = str(should_flag).lower() in ['true', '1', 'yes']
                flag_result["should_flag"] = should_flag
            
            state["flag_decision"] = flag_result
            
            logger.info(f"Flag decision: {flag_result.get('should_flag', False)}")
            
        except Exception as e:
            logger.error(f"Flag decision failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            state["flag_decision"] = {
                "should_flag": True,  # Default to flagging on error
                "flag_reasons": ["Analysis error - defaulting to human review"],
                "urgency_level": "medium",
                "reasoning": "Decision failed, defaulting to human review for safety",
                "confidence": 0.5
            }
        
        return state
    
    def _decide_response(self, state: ClientMessageState) -> ClientMessageState:
        """Decide whether and how to respond using OpenAI reasoning model with conversation awareness"""
        
        client_context = self._build_client_context(state["client_details"])
        conversation_context = self._build_conversation_context(state["conversation_history"])
        sentiment_data = state["sentiment_analysis"]
        flag_decision = state["flag_decision"]
        
        # Analyze response patterns and client preferences
        conversation_insights = ""
        if state["conversation_history"]:
            client_messages = [msg for msg in state["conversation_history"] if msg.get("sender") == "client"]
            system_responses = [msg for msg in state["conversation_history"] if msg.get("sender") == "system"]
            
            # Analyze client's communication style
            recent_client_msgs = client_messages[-3:]
            avg_length = sum(len(msg.get("content", "")) for msg in recent_client_msgs) / max(len(recent_client_msgs), 1)
            
            conversation_insights = f"""
            RESPONSE CONTEXT:
            - Client's typical message length: {'Long' if avg_length > 100 else 'Short'}
            - Recent system responses: {len(system_responses)}
            - Client engagement level: {'High' if len(client_messages) > 5 else 'Normal'}
            - Conversation pattern: {sentiment_data.get('pattern_change', 'consistent')}
            """
        
        response_prompt = f"""
        You are an expert communication agent for a law firm's AI system.
        
        CLIENT CONTEXT:
        {client_context}
        
        CONVERSATION HISTORY AND PATTERNS:
        {conversation_context}
        
        {conversation_insights}
        
        CURRENT MESSAGE:
        "{state['message']}"
        
        SENTIMENT ANALYSIS:
        {json.dumps(sentiment_data, indent=2)}
        
        CONCERN LEVEL: {state['concern_level']}
        
        FLAG DECISION:
        {json.dumps(flag_decision, indent=2)}
        
        MESSAGE COUNT: {state['message_count']}
        
        RESPONSE DECISION CRITERIA:
        
        DO NOT RESPOND IF:
        - Message is flagged for human review
        - Message is simple acknowledgment ("ok", "thanks", "got it")
        - Message count exceeds 25 for the week
        - Client requested to pause messages
        - Recent escalation pattern suggests human intervention needed
        
        RESPOND IF:
        - Simple question that doesn't require legal advice
        - Client needs reassurance or empathy
        - Routine check-in response
        - Clarification needed for basic information
        - Acknowledgment of client's concern while awaiting human response
        
        RESPONSE GUIDELINES:
        - Match client's communication style and energy from conversation history
        - Be empathetic and human-like
        - Keep responses concise but caring
        - Never provide legal or medical advice
        - Use client's name and case manager's name naturally
        - Show personality but stay professional
        - Reference previous conversation context when appropriate
        - Acknowledge patterns (e.g., "I understand you've been asking about...")
        
        IMPORTANT: Use conversation history to personalize response - if client was previously positive, acknowledge the change. If they've asked similar questions, acknowledge their continued concern.
        
        Provide your response decision in JSON format:
        {{
            "should_respond": true|false,
            "response_type": "empathetic|informational|clarification|acknowledgment|ignore",
            "tone": "casual|professional|supportive|brief",
            "conversation_context": "How conversation history should influence the response",
            "reasoning": "Your detailed reasoning considering conversation patterns and client relationship",
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = self.reasoning_model.invoke([
                SystemMessage(content="You are an expert communication agent. Balance helpfulness with appropriate boundaries."),
                HumanMessage(content=response_prompt)
            ])
            
            response_result = self._parse_json_response(response.content)
            
            # Ensure response_result is a dictionary
            if not isinstance(response_result, dict):
                logger.warning(f"Response result is not a dict: {type(response_result)}")
                response_result = {}
            
            state["response_decision"] = response_result
            
            logger.info(f"Response decision: {response_result.get('should_respond', False)}")
            
        except Exception as e:
            logger.error(f"Response decision failed: {e}")
            state["response_decision"] = {
                "should_respond": False,  # Default to not responding on error
                "response_type": "ignore",
                "tone": "professional",
                "reasoning": "Decision failed, defaulting to human review",
                "confidence": 0.5
            }
        
        return state
    
    def _generate_response_content(self, state: ClientMessageState) -> ClientMessageState:
        """Generate the actual response content when the system decides to respond"""
        
        response_decision = state["response_decision"]
        
        # Only generate response if we should respond
        if not response_decision.get("should_respond", False):
            state["response_content"] = None
            return state
            
        client_context = self._build_client_context(state["client_details"])
        conversation_context = self._build_conversation_context(state["conversation_history"])
        sentiment_data = state["sentiment_analysis"]
        
        response_type = response_decision.get("response_type", "acknowledgment")
        tone = response_decision.get("tone", "professional")
        
        response_prompt = f"""
        You are a friendly and professional AI assistant for a law firm. Generate a personalized response to the client.
        
        CLIENT CONTEXT:
        {client_context}
        
        CONVERSATION HISTORY:
        {conversation_context}
        
        CURRENT MESSAGE:
        "{state['message']}"
        
        SENTIMENT: {sentiment_data.get('sentiment', 'neutral')}
        RESPONSE TYPE: {response_type}
        TONE: {tone}
        
        RESPONSE GUIDELINES:
        - Be warm, empathetic, and professional
        - Use the client's name ({state['client_details'].get('name', 'there')})
        - Reference their case manager ({', '.join(state['client_details'].get('managing_users', ['your case manager']))}) when appropriate
        - Keep it concise (2-3 sentences max)
        - Show genuine care and understanding
        - Never provide legal or medical advice
        - If this is their first message, welcome them warmly
        - If they seem concerned, acknowledge their feelings
        - End with an invitation for them to share more or ask questions
        
        IMPORTANT BOUNDARIES:
        - Do not give legal advice or case valuations
        - Do not make promises about outcomes
        - Do not provide medical recommendations
        - Do refer complex matters to their legal team
        
        Generate only the response text (no JSON, no quotes, just the message):
        """
        
        try:
            response = self.chat_model.invoke([
                SystemMessage(content="You are a caring, professional AI assistant for a law firm. Generate warm, helpful responses that stay within appropriate boundaries."),
                HumanMessage(content=response_prompt)
            ])
            
            # Clean up the response content
            response_content = response.content.strip()
            
            # Remove any quotes or JSON formatting that might have been added
            if response_content.startswith('"') and response_content.endswith('"'):
                response_content = response_content[1:-1]
            
            state["response_content"] = response_content
            logger.info(f"Generated response content: {response_content[:50]}...")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Fallback response
            client_name = state['client_details'].get('name', 'there')
            state["response_content"] = f"Hi {client_name}! Thank you for reaching out. I'm here to help with any questions you might have about your case. What can I assist you with today?"
        
        return state
    
    def _compile_results(self, state: ClientMessageState) -> ClientMessageState:
        """Compile final results in the required JSON format"""
        
        sentiment_data = state["sentiment_analysis"]
        flag_decision = state["flag_decision"]
        response_decision = state["response_decision"]
        
        # Determine action based on decisions
        if flag_decision.get("should_flag", False):
            action = "flag"
        elif response_decision.get("should_respond", False):
            action = "respond"
        else:
            action = "ignore"
        
        # Compile final result
        final_result = {
            "action": action,
            "should_respond": response_decision.get("should_respond", False),
            "should_flag": flag_decision.get("should_flag", False),
            "concern_level": state["concern_level"],
            "sentiment": sentiment_data.get("sentiment", "neutral"),
            "reasoning": self._compile_reasoning(sentiment_data, flag_decision, response_decision),
            "confidence": min(
                sentiment_data.get("confidence", 0.5),
                flag_decision.get("confidence", 0.5),
                response_decision.get("confidence", 0.5)
            ),
            "response_content": state.get("response_content")  # Include the actual response content
        }
        
        state["final_result"] = final_result
        
        logger.info(f"Final result compiled: {action}")
        
        return state
    
    def _compile_reasoning(self, sentiment_data: Dict, flag_decision: Dict, response_decision: Dict) -> str:
        """Compile reasoning from all decision points with conversation context"""
        
        reasoning_parts = []
        
        # Sentiment reasoning with pattern context
        if sentiment_data.get("reasoning"):
            reasoning_parts.append(f"Sentiment Analysis: {sentiment_data['reasoning']}")
        
        # Add pattern context if available
        if sentiment_data.get("pattern_change"):
            reasoning_parts.append(f"Pattern Change: {sentiment_data['pattern_change']}")
        
        # Flag reasoning with historical context
        if flag_decision.get("reasoning"):
            reasoning_parts.append(f"Flag Decision: {flag_decision['reasoning']}")
        
        # Add pattern context for flagging if available
        if flag_decision.get("pattern_context"):
            reasoning_parts.append(f"Pattern Context: {flag_decision['pattern_context']}")
        
        # Response reasoning with conversation context
        if response_decision.get("reasoning"):
            reasoning_parts.append(f"Response Decision: {response_decision['reasoning']}")
        
        # Add conversation context for response if available
        if response_decision.get("conversation_context"):
            reasoning_parts.append(f"Conversation Context: {response_decision['conversation_context']}")
        
        return " | ".join(reasoning_parts)
    
    def _build_client_context(self, client_details: Dict) -> str:
        """Build client context string for prompts with enhanced insights"""
        
        context_parts = []
        
        if client_details.get("name"):
            context_parts.append(f"Name: {client_details['name']}")
        
        if client_details.get("gender"):
            context_parts.append(f"Gender: {client_details['gender']}")
        
        # Enhanced timeline analysis
        if client_details.get("incident_date"):
            days_since_incident = self._calculate_days_since(client_details["incident_date"])
            context_parts.append(f"Days since incident: {days_since_incident}")
            
            # Add contextual insight based on timeline
            if days_since_incident < 30:
                context_parts.append("- Timeline context: Fresh incident, likely high emotional state")
            elif days_since_incident < 90:
                context_parts.append("- Timeline context: Early case phase, establishing trust period")
            elif days_since_incident < 365:
                context_parts.append("- Timeline context: Active case development, may have timeline concerns")
            else:
                context_parts.append("- Timeline context: Long-term case, possible frustration with delays")
        
        if client_details.get("signup_date"):
            days_since_signup = self._calculate_days_since(client_details["signup_date"])
            context_parts.append(f"Days since signup: {days_since_signup}")
            
            # Add relationship context
            if days_since_signup < 7:
                context_parts.append("- Relationship context: New client, onboarding phase")
            elif days_since_signup < 30:
                context_parts.append("- Relationship context: Establishing relationship, building trust")
            elif days_since_signup < 90:
                context_parts.append("- Relationship context: Active engagement period")
            else:
                context_parts.append("- Relationship context: Established client, expect familiarity")
        
        if client_details.get("lawyer_name"):
            context_parts.append(f"Lawyer: {client_details['lawyer_name']}")
        
        if client_details.get("managing_users"):
            context_parts.append(f"Case Manager: {', '.join(client_details['managing_users'])}")
        
        if client_details.get("injuries"):
            context_parts.append(f"Injuries: {client_details['injuries']}")
        
        if client_details.get("case_info"):
            context_parts.append(f"Case Info: {client_details['case_info']}")
        
        return "\n".join(context_parts)

    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        """Build conversation context string for prompts with enhanced pattern analysis"""
        
        if not conversation_history:
            return "No previous conversation history"
        
        context_parts = []
        
        # Recent conversation (last 5 messages)
        context_parts.append("RECENT CONVERSATION:")
        for i, message in enumerate(conversation_history[-5:]):
            sender = message.get("sender", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            
            context_parts.append(f"{i+1}. {sender} ({timestamp}): {content}")
        
        # Conversation patterns analysis
        if len(conversation_history) >= 3:
            context_parts.append("\nCONVERSATION PATTERNS:")
            
            # Analyze message frequency and gaps
            client_messages = [msg for msg in conversation_history if msg.get("sender") == "client"]
            if client_messages:
                recent_client_msgs = client_messages[-10:]  # Last 10 client messages
                
                # Analyze sentiment progression
                positive_keywords = ["thank", "good", "great", "appreciate", "happy", "satisfied"]
                negative_keywords = ["angry", "frustrated", "upset", "disappointed", "worried", "concerned"]
                
                recent_sentiment_trend = []
                for msg in recent_client_msgs:
                    content_lower = msg.get("content", "").lower()
                    if any(word in content_lower for word in positive_keywords):
                        recent_sentiment_trend.append("positive")
                    elif any(word in content_lower for word in negative_keywords):
                        recent_sentiment_trend.append("negative")
                    else:
                        recent_sentiment_trend.append("neutral")
                
                if recent_sentiment_trend:
                    context_parts.append(f"- Recent sentiment trend: {' → '.join(recent_sentiment_trend[-3:])}")
                
                # Analyze topic consistency
                legal_topics = ["settlement", "lawyer", "attorney", "case", "court", "insurance"]
                medical_topics = ["doctor", "treatment", "pain", "injury", "medical", "hospital"]
                timeline_topics = ["when", "how long", "timeline", "status", "update"]
                
                topic_mentions = {"legal": 0, "medical": 0, "timeline": 0}
                for msg in recent_client_msgs:
                    content_lower = msg.get("content", "").lower()
                    if any(word in content_lower for word in legal_topics):
                        topic_mentions["legal"] += 1
                    if any(word in content_lower for word in medical_topics):
                        topic_mentions["medical"] += 1
                    if any(word in content_lower for word in timeline_topics):
                        topic_mentions["timeline"] += 1
                
                dominant_topics = [topic for topic, count in topic_mentions.items() if count > 1]
                if dominant_topics:
                    context_parts.append(f"- Recurring topics: {', '.join(dominant_topics)}")
                
                # Analyze response patterns
                system_responses = [msg for msg in conversation_history if msg.get("sender") == "system"]
                if system_responses:
                    context_parts.append(f"- Previous system responses: {len(system_responses)}")
                    
                    # Check for flagged messages indicator
                    flagged_indicators = ["forwarded", "escalated", "manager", "attorney", "review"]
                    recent_flags = 0
                    for msg in system_responses[-5:]:
                        content_lower = msg.get("content", "").lower()
                        if any(word in content_lower for word in flagged_indicators):
                            recent_flags += 1
                    
                    if recent_flags > 0:
                        context_parts.append(f"- Recent escalations/flags: {recent_flags}")
        
        return "\n".join(context_parts)

    def _calculate_days_since(self, date_str: str) -> int:
        """Calculate days since a given date"""
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            days_diff = (datetime.now() - date_obj).days
            return days_diff
        except:
            return 0
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response from OpenAI, handling potential formatting issues"""
        
        # Handle non-string inputs
        if not isinstance(response, str):
            logger.warning(f"Response is not a string: {type(response)}")
            if hasattr(response, 'content'):
                response = response.content
            else:
                response = str(response)
        
        response = response.strip()
        
        try:
            # First try to parse the entire response as JSON
            parsed = json.loads(response)
            # Ensure we always return a dictionary
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    logger.warning("JSON response was a list, taking first dict element")
                    return parsed[0]
                else:
                    logger.warning(f"JSON list contains non-dict: {type(parsed[0])}")
                    return {}
            else:
                logger.warning(f"Unexpected JSON type: {type(parsed)}")
                return {}
        except json.JSONDecodeError:
            pass
        
        # Look for JSON in code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        return parsed
                    elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                        return parsed[0]
                except json.JSONDecodeError:
                    pass
        
        try:
            # Look for JSON array first
            array_start = response.find('[')
            array_end = response.rfind(']') + 1
            
            if array_start != -1 and array_end != -1:
                json_str = response[array_start:array_end]
                parsed = json.loads(json_str)
                # If it's a list, return the first dict item
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], dict):
                        logger.warning("Found JSON array, taking first dict element")
                        return parsed[0]
                    else:
                        logger.warning(f"JSON array contains non-dict: {type(parsed[0])}")
                        return {}
                return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
            
        try:
            # Look for JSON object
            obj_start = response.find('{')
            obj_end = response.rfind('}') + 1
            
            if obj_start != -1 and obj_end != -1:
                json_str = response[obj_start:obj_end]
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    logger.warning(f"Parsed object is not a dict: {type(parsed)}")
                    return {}
        except json.JSONDecodeError:
            pass
        
        # Log the raw response for debugging
        logger.warning(f"Failed to parse JSON from response: {response[:500]}...")
        return {}
    
    def process_client_message(self, 
                             message: str, 
                             client_details: Dict,
                             message_count: int = 0,
                             enable_human_delay: bool = True,
                             conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Main function to process client messages using agentic AI
        
        Args:
            message: The client message to process
            client_details: Dictionary containing client information
            message_count: Current message count in conversation
            enable_human_delay: Whether to add human-like delay
            conversation_history: Previous conversation messages
        
        Returns:
            Dictionary containing processing results
        """
        
        logger.info(f"Processing message from {client_details.get('name', 'unknown')}")
        
        # Initialize state
        initial_state = {
            "message": message,
            "client_details": client_details,
            "message_count": message_count,
            "enable_human_delay": enable_human_delay,
            "conversation_history": conversation_history or [],
            "sentiment_analysis": None,
            "concern_level": None,
            "flag_decision": None,
            "response_decision": None,
            "response_content": None,
            "final_result": None,
            "messages": []
        }
        
        # Add human-like delay if enabled
        if enable_human_delay:
            delay = random.uniform(2, 8)  # Random delay between 2-8 seconds
            logger.info(f"Adding human-like delay: {delay:.1f} seconds")
            time.sleep(delay)
        
        try:
            # Process through the graph
            result = self.graph.invoke(initial_state)
            
            # Return the final result
            return result["final_result"]
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            
            # Return safe default result
            return {
                "action": "flag",
                "should_respond": False,
                "should_flag": True,
                "concern_level": "high",
                "sentiment": "neutral",
                "reasoning": f"Processing failed: {str(e)} - Defaulting to human review",
                "confidence": 0.1,
                "response_content": None
            }
    
    def generate_client_insights(self, 
                               client_details: Dict,
                               conversation_history: List[Dict],
                               time_period_days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate actionable insights about client behavior and communication patterns
        
        Args:
            client_details: Dictionary containing client information
            conversation_history: Complete conversation history
            time_period_days: Number of days to analyze (default 30)
        
        Returns:
            List of insight dictionaries containing:
                - insight_type: "positive", "concern", "action_required"
                - category: "satisfaction", "communication", "case_progress", "retention_risk"
                - message: Human-readable insight message
                - date: Date of the insight
                - status: "new", "reviewed", "action_required"
                - confidence: Confidence score (0.0-1.0)
                - supporting_evidence: List of supporting messages/events
        """
        
        logger.info(f"Generating insights for {client_details.get('name', 'unknown')}")
        
        try:
            # Filter conversation history by time period
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            recent_history = []
            
            for msg in conversation_history:
                try:
                    msg_date = datetime.strptime(msg.get("timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    if msg_date >= cutoff_date:
                        recent_history.append(msg)
                except:
                    recent_history.append(msg)  # Include if date parsing fails
            
            # Build context for insight generation
            client_context = self._build_client_context(client_details)
            conversation_context = self._build_conversation_context(recent_history)
            
            # Analyze conversation patterns for insights
            insight_analysis = self._analyze_conversation_for_insights(recent_history, client_details)
            
            insight_prompt = f"""
            Analyze the client communication and generate specific insights. Based on the conversation data provided, create actionable insights for the law firm.
            
            CLIENT INFORMATION:
            {client_context}
            
            RECENT CONVERSATIONS ({time_period_days} days):
            {conversation_context}
            
            ANALYSIS DATA:
            {insight_analysis}
            
            TASK: Generate 2-4 specific insights based on the available data. Focus on what can be determined from the actual conversation content.
            
            INSIGHT CATEGORIES:
            1. Communication patterns and preferences
            2. Client satisfaction indicators  
            3. Case progress concerns or questions
            4. Potential retention risks or positive indicators
            
            OUTPUT FORMAT: Return ONLY a valid JSON array with this exact structure:
            [
                {{
                    "insight_type": "positive",
                    "category": "communication",
                    "message": "Client shows positive engagement with initial contact and responds well to warm, personal communication style",
                    "date": "{datetime.now().strftime('%Y-%m-%d')}",
                    "status": "new",
                    "confidence": 0.8,
                    "supporting_evidence": ["Client responded positively to greeting", "Engaged in conversation"],
                    "recommended_actions": ["Continue warm communication approach", "Maintain personal touch"],
                    "priority": "low"
                }}
            ]
            
            IMPORTANT: 
            - Return ONLY the JSON array, no other text
            - Each insight must have a clear, specific message
            - Base insights on actual conversation content
            - Include realistic confidence scores (0.0-1.0)
            - Provide specific supporting evidence from the conversation
            - Suggest actionable recommendations
            """
            
            response = self.reasoning_model.invoke([
                SystemMessage(content="You are an expert client relationship analyst. Generate clear, actionable insights."),
                HumanMessage(content=insight_prompt)
            ])
            
            # Parse the response
            insights_data = self._parse_json_response(response.content)
            
            # If response is a single dict, wrap in array
            if isinstance(insights_data, dict):
                insights_data = [insights_data]
            elif not isinstance(insights_data, list):
                insights_data = []
            
            # Validate and format insights
            formatted_insights = []
            for insight in insights_data:
                if isinstance(insight, dict) and insight.get("message"):
                    formatted_insight = {
                        "insight_type": insight.get("insight_type", "concern"),
                        "category": insight.get("category", "communication"),
                        "message": insight.get("message", ""),
                        "date": insight.get("date", datetime.now().strftime('%Y-%m-%d')),
                        "status": insight.get("status", "new"),
                        "confidence": float(insight.get("confidence", 0.7)),
                        "supporting_evidence": insight.get("supporting_evidence", []),
                        "recommended_actions": insight.get("recommended_actions", []),
                        "priority": insight.get("priority", "medium")
                    }
                    formatted_insights.append(formatted_insight)
            
            logger.info(f"Generated {len(formatted_insights)} insights")
            return formatted_insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            
            # Generate fallback insights based on available data
            fallback_insights = []
            
            # If we have some conversation data, generate basic insights
            if conversation_history:
                client_messages = [msg for msg in conversation_history if msg.get("sender") == "client"]
                system_messages = [msg for msg in conversation_history if msg.get("sender") == "system"]
                
                # Basic communication insight
                if len(client_messages) > 0:
                    fallback_insights.append({
                        "insight_type": "positive",
                        "category": "communication",
                        "message": f"Client has initiated {len(client_messages)} conversation(s), showing active engagement with the firm.",
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "status": "new",
                        "confidence": 0.7,
                        "supporting_evidence": [f"Client sent {len(client_messages)} messages"],
                        "recommended_actions": ["Continue responsive communication", "Monitor for any concerns"],
                        "priority": "low"
                    })
                
                # Timeline insight based on case stage
                days_since_incident = self._calculate_days_since(client_details.get("incident_date", ""))
                if days_since_incident > 90:
                    fallback_insights.append({
                        "insight_type": "concern",
                        "category": "case_progress",
                        "message": f"Case is {days_since_incident} days old. Client may have questions about timeline and progress.",
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "status": "new",
                        "confidence": 0.6,
                        "supporting_evidence": [f"Incident occurred {days_since_incident} days ago"],
                        "recommended_actions": ["Provide case progress update", "Set clear timeline expectations"],
                        "priority": "medium"
                    })
            
            # If no insights generated, provide a basic one
            if not fallback_insights:
                fallback_insights.append({
                    "insight_type": "concern",
                    "category": "communication",
                    "message": "Limited conversation data available. Recommend reaching out to client to establish communication.",
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "status": "new",
                    "confidence": 0.5,
                    "supporting_evidence": ["Minimal conversation history"],
                    "recommended_actions": ["Initiate client contact", "Establish regular communication schedule"],
                    "priority": "medium"
                })
            
            return fallback_insights
    
    def _analyze_conversation_for_insights(self, conversation_history: List[Dict], client_details: Dict) -> str:
        """Analyze conversation patterns to provide context for insight generation"""
        
        if not conversation_history:
            return "No conversation history available for analysis"
        
        analysis_parts = []
        
        # Message frequency analysis
        client_messages = [msg for msg in conversation_history if msg.get("sender") == "client"]
        system_messages = [msg for msg in conversation_history if msg.get("sender") == "system"]
        
        analysis_parts.append(f"MESSAGE FREQUENCY:")
        analysis_parts.append(f"- Client messages: {len(client_messages)}")
        analysis_parts.append(f"- System responses: {len(system_messages)}")
        analysis_parts.append(f"- Response ratio: {len(system_messages)/max(len(client_messages), 1):.2f}")
        
        # Sentiment progression analysis
        positive_keywords = ["thank", "good", "great", "appreciate", "happy", "satisfied", "excellent"]
        negative_keywords = ["angry", "frustrated", "upset", "disappointed", "worried", "concerned", "unhappy"]
        
        sentiment_timeline = []
        for msg in client_messages:
            content_lower = msg.get("content", "").lower()
            if any(word in content_lower for word in positive_keywords):
                sentiment_timeline.append("positive")
            elif any(word in content_lower for word in negative_keywords):
                sentiment_timeline.append("negative")
            else:
                sentiment_timeline.append("neutral")
        
        if sentiment_timeline:
            analysis_parts.append(f"\nSENTIMENT PROGRESSION:")
            analysis_parts.append(f"- Overall sentiment: {' → '.join(sentiment_timeline)}")
            
            # Sentiment trend
            recent_sentiments = sentiment_timeline[-5:]
            positive_count = recent_sentiments.count("positive")
            negative_count = recent_sentiments.count("negative")
            
            if positive_count > negative_count:
                analysis_parts.append(f"- Recent trend: Positive ({positive_count}/{len(recent_sentiments)})")
            elif negative_count > positive_count:
                analysis_parts.append(f"- Recent trend: Negative ({negative_count}/{len(recent_sentiments)})")
            else:
                analysis_parts.append(f"- Recent trend: Neutral/Mixed")
        
        # Topic analysis
        topic_keywords = {
            "timeline": ["when", "how long", "timeline", "status", "update", "progress"],
            "legal_advice": ["lawyer", "attorney", "legal", "case", "settlement", "court"],
            "medical": ["doctor", "treatment", "pain", "injury", "medical", "hospital"],
            "financial": ["money", "cost", "payment", "bill", "insurance", "coverage"],
            "communication": ["call", "email", "message", "contact", "response", "update"]
        }
        
        topic_frequency = {topic: 0 for topic in topic_keywords.keys()}
        
        for msg in client_messages:
            content_lower = msg.get("content", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(word in content_lower for word in keywords):
                    topic_frequency[topic] += 1
        
        top_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        
        analysis_parts.append(f"\nTOP DISCUSSION TOPICS:")
        for topic, count in top_topics:
            if count > 0:
                analysis_parts.append(f"- {topic.title()}: {count} mentions")
        
        # Response pattern analysis
        flagged_indicators = ["forwarded", "escalated", "manager", "attorney", "review", "flagged"]
        escalation_count = 0
        
        for msg in system_messages:
            content_lower = msg.get("content", "").lower()
            if any(word in content_lower for word in flagged_indicators):
                escalation_count += 1
        
        analysis_parts.append(f"\nESCALATION PATTERNS:")
        analysis_parts.append(f"- Messages escalated: {escalation_count}")
        analysis_parts.append(f"- Escalation rate: {escalation_count/max(len(client_messages), 1):.2f}")
        
        # Timeline context
        days_since_incident = self._calculate_days_since(client_details.get("incident_date", ""))
        days_since_signup = self._calculate_days_since(client_details.get("signup_date", ""))
        
        analysis_parts.append(f"\nTIMELINE CONTEXT:")
        analysis_parts.append(f"- Days since incident: {days_since_incident}")
        analysis_parts.append(f"- Days since signup: {days_since_signup}")
        analysis_parts.append(f"- Case stage: {self._determine_case_stage(days_since_incident, days_since_signup)}")
        
        return "\n".join(analysis_parts)
    
    def _determine_case_stage(self, days_since_incident: int, days_since_signup: int) -> str:
        """Determine the current stage of the case based on timeline"""
        
        if days_since_signup < 7:
            return "Initial onboarding"
        elif days_since_signup < 30:
            return "Early relationship building"
        elif days_since_incident < 90:
            return "Active case development"
        elif days_since_incident < 365:
            return "Case progression"
        else:
            return "Long-term case management"
    
# Export the main function for external use
def process_client_message(message: str, 
                         client_details: dict,
                         message_count: int = 0,
                         enable_human_delay: bool = True,
                         conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Process a client message using the Arviso AI Agent
    
    Args:
        message: The client message to process
        client_details: Dictionary containing:
            - name: Client name
            - gender: Client gender
            - incident_date: Date of incident (YYYY-MM-DD)
            - signup_date: Date client signed up (YYYY-MM-DD)
            - managing_users: List of managing users
            - lawyer_name: Assigned lawyer
            - injuries: Description of injuries
            - case_info: Case details
        message_count: Current message count in conversation
        enable_human_delay: Whether to add human-like delay
        conversation_history: Previous conversation messages
    
    Returns:
        Dictionary containing:
            - action: "respond", "flag", or "ignore"
            - should_respond: Boolean
            - should_flag: Boolean
            - concern_level: "low", "medium", or "high"
            - sentiment: "positive", "neutral", or "negative"
            - reasoning: Detailed reasoning for decisions
            - confidence: Confidence score (0.0-1.0)
    """
    
    # Initialize the AI agent
    agent = ArvisoAIAgent()
    
    # Process the message
    return agent.process_client_message(
        message=message,
        client_details=client_details,
        message_count=message_count,
        enable_human_delay=enable_human_delay,
        conversation_history=conversation_history
    )
