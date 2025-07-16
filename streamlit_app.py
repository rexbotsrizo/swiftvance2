"""
Streamlit Frontend for Arviso AI Agent Testing
Provides interface to test client message processing and insights generation
"""

import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from client_message_processor import ArvisoAIAgent, process_client_message

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Arviso AI Agent Testing",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-positive {
        border-left: 4px solid #28a745;
        background-color: #d4edda;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .insight-concern {
        border-left: 4px solid #ffc107;
        background-color: #fff3cd;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .insight-action {
        border-left: 4px solid #dc3545;
        background-color: #f8d7da;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'insights' not in st.session_state:
    st.session_state.insights = []

# Main title
st.markdown('<div class="main-header">🤖 Arviso AI Agent Testing Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input with .env fallback
    env_api_key = os.getenv("OPENAI_API_KEY", "")
    
    if env_api_key:
        st.success("✅ API Key loaded from .env file")
        api_key = env_api_key
        st.text_input("OpenAI API Key", value="***************************", disabled=True, help="API key loaded from .env file")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key or add it to .env file")
    
    if api_key:
        try:
            if st.session_state.agent is None:
                st.session_state.agent = ArvisoAIAgent(openai_api_key=api_key)
            st.success("✅ AI Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing agent: {str(e)}")
    else:
        st.warning("⚠️ Please provide OpenAI API Key")
    
    st.divider()
    
    # Client Details
    st.header("👤 Client Information")
    
    client_name = st.text_input("Client Name", value="John Doe")
    client_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    incident_date = st.date_input("Incident Date", value=datetime.now() - timedelta(days=60))
    signup_date = st.date_input("Signup Date", value=datetime.now() - timedelta(days=30))
    lawyer_name = st.text_input("Lawyer Name", value="Sarah Johnson")
    case_manager = st.text_input("Case Manager", value="Mike Smith")
    injuries = st.text_area("Injuries", value="Back injury, whiplash")
    case_info = st.text_area("Case Info", value="Auto accident on highway")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation History"):
        st.session_state.conversation_history = []
        st.session_state.processing_results = []
        st.session_state.insights = []
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["💬 Message Processing", "📊 Insights Dashboard", "📈 Analytics", "🔍 History"])

with tab1:
    st.header("Message Processing & Testing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Test Message")
        
        # Quick test examples
        st.write("**Quick Test Examples:**")
        example_messages = {
            "Positive": "Thank you so much for the update! I really appreciate how you've been handling my case.",
            "Concern": "I'm getting worried about the timeline. It's been 3 months and I haven't heard much progress.",
            "Flag Required": "I think I need to speak with another lawyer. This isn't working out.",
            "Medical Question": "Should I go back to the doctor for my back pain?",
            "Legal Advice": "What do you think my case is worth? Can we settle for $50,000?",
            "Routine Check": "Hi, just checking in to see if there are any updates on my case."
        }
        
        selected_example = st.selectbox("Select example message:", ["Custom"] + list(example_messages.keys()))
        
        if selected_example != "Custom":
            default_message = example_messages[selected_example]
        else:
            default_message = ""
        
        test_message = st.text_area("Enter client message:", value=default_message, height=100)
        
        col_a, col_b = st.columns(2)
        with col_a:
            enable_delay = st.checkbox("Enable human-like delay", value=True)
        with col_b:
            message_count = st.number_input("Message count", min_value=0, value=len(st.session_state.conversation_history))
        
        if st.button("🚀 Process Message", type="primary"):
            if st.session_state.agent is None:
                st.error("Please ensure your OpenAI API key is configured!")
            elif not test_message.strip():
                st.error("Please enter a message to process!")
            else:
                # Prepare client details
                client_details = {
                    "name": client_name,
                    "gender": client_gender,
                    "incident_date": incident_date.strftime("%Y-%m-%d"),
                    "signup_date": signup_date.strftime("%Y-%m-%d"),
                    "lawyer_name": lawyer_name,
                    "managing_users": [case_manager],
                    "injuries": injuries,
                    "case_info": case_info
                }
                
                # Process message
                with st.spinner("Processing message..."):
                    try:
                        result = st.session_state.agent.process_client_message(
                            message=test_message,
                            client_details=client_details,
                            message_count=message_count,
                            enable_human_delay=enable_delay,
                            conversation_history=st.session_state.conversation_history
                        )
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "sender": "client",
                            "content": test_message,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Add system response if applicable with actual content
                        if result.get("should_respond") and result.get("response_content"):
                            st.session_state.conversation_history.append({
                                "sender": "system",
                                "content": result.get("response_content"),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        
                        # Store result
                        st.session_state.processing_results.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "message": test_message,
                            "result": result
                        })
                        
                        st.success("✅ Message processed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing message: {str(e)}")
    
    with col2:
        st.subheader("🎯 Processing Results")
        
        if st.session_state.processing_results:
            latest_result = st.session_state.processing_results[-1]["result"]
            
            # Action badge
            action = latest_result.get("action", "unknown")
            action_colors = {
                "respond": "🟢",
                "flag": "🔴", 
                "ignore": "⚪"
            }
            
            st.markdown(f"**Action:** {action_colors.get(action, '❓')} {action.upper()}")
            
            # Metrics
            col_i, col_ii = st.columns(2)
            with col_i:
                sentiment = latest_result.get("sentiment", "neutral")
                sentiment_colors = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
                st.metric("Sentiment", sentiment, delta=None)
            
            with col_ii:
                concern = latest_result.get("concern_level", "medium")
                st.metric("Concern Level", concern.upper())
            
            # Confidence
            confidence = latest_result.get("confidence", 0.0)
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Flags
            st.markdown("**Decisions:**")
            st.write(f"• Should Respond: {'✅' if latest_result.get('should_respond') else '❌'}")
            st.write(f"• Should Flag: {'🚩' if latest_result.get('should_flag') else '✅'}")
            
            # Reasoning
            with st.expander("📝 Detailed Reasoning"):
                st.write(latest_result.get("reasoning", "No reasoning provided"))
            
            # Show actual response content if available
            if latest_result.get("response_content"):
                with st.expander("💬 AI Response Content"):
                    st.info("📧 **Response to Client:**")
                    st.write(f'"{latest_result.get("response_content")}"')
                    st.caption("This is the actual message that would be sent to the client.")
        
        else:
            st.info("Process a message to see results here")

with tab2:
    st.header("📊 Client Insights Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Generate Insights")
        
        time_period = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
        
        if st.button("📈 Generate Client Insights", type="primary"):
            if st.session_state.agent is None:
                st.error("Please ensure your OpenAI API key is configured!")
            elif not st.session_state.conversation_history:
                st.error("No conversation history available for analysis!")
            else:
                client_details = {
                    "name": client_name,
                    "gender": client_gender,
                    "incident_date": incident_date.strftime("%Y-%m-%d"),
                    "signup_date": signup_date.strftime("%Y-%m-%d"),
                    "lawyer_name": lawyer_name,
                    "managing_users": [case_manager],
                    "injuries": injuries,
                    "case_info": case_info
                }
                
                with st.spinner("Generating insights..."):
                    try:
                        st.info(f"Analyzing {len(st.session_state.conversation_history)} messages over {time_period} days...")
                        
                        # Debug: Show conversation history structure
                        with st.expander("🔍 Debug: Conversation Data"):
                            st.write("Conversation History:")
                            for i, msg in enumerate(st.session_state.conversation_history):
                                st.write(f"{i+1}. {msg.get('sender', 'unknown')}: {msg.get('content', '')[:50]}...")
                        
                        insights = st.session_state.agent.generate_client_insights(
                            client_details=client_details,
                            conversation_history=st.session_state.conversation_history,
                            time_period_days=time_period
                        )
                        
                        st.write(f"Debug: Raw insights returned: {type(insights)} with {len(insights) if insights else 0} items")
                        
                        if insights and len(insights) > 0:
                            st.session_state.insights = insights
                            st.success(f"✅ Generated {len(insights)} insights!")
                            st.rerun()  # Refresh to show insights
                        else:
                            st.warning("⚠️ No insights generated. This might be due to insufficient conversation data.")
                            st.info("Try adding more conversation messages and then generate insights again.")
                            st.session_state.insights = []
                        
                    except Exception as e:
                        st.error(f"❌ Error generating insights: {str(e)}")
                        st.error("Please check the logs for more details.")
                        # Show more detailed error for debugging
                        with st.expander("🔍 Error Details"):
                            st.code(str(e))
                            import traceback
                            st.code(traceback.format_exc())
    
    with col2:
        st.subheader("📋 Insight Summary")
        
        if st.session_state.insights:
            insights = st.session_state.insights
            
            # Count by type
            type_counts = {}
            for insight in insights:
                itype = insight.get("insight_type", "unknown")
                type_counts[itype] = type_counts.get(itype, 0) + 1
            
            for itype, count in type_counts.items():
                emoji = {"positive": "🟢", "concern": "🟡", "action_required": "🔴"}.get(itype, "❓")
                st.metric(f"{emoji} {itype.title().replace('_', ' ')}", count)
        
        else:
            st.info("Generate insights to see summary")
    
    # Display insights
    if st.session_state.insights:
        st.subheader("💡 Generated Insights")
        
        # Add filter options
        if len(st.session_state.insights) > 1:
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                filter_type = st.selectbox("Filter by Type", ["All"] + list(set([i.get("insight_type", "unknown") for i in st.session_state.insights])))
            with col_filter2:
                filter_category = st.selectbox("Filter by Category", ["All"] + list(set([i.get("category", "unknown") for i in st.session_state.insights])))
            
            # Apply filters
            filtered_insights = st.session_state.insights
            if filter_type != "All":
                filtered_insights = [i for i in filtered_insights if i.get("insight_type") == filter_type]
            if filter_category != "All":
                filtered_insights = [i for i in filtered_insights if i.get("category") == filter_category]
        else:
            filtered_insights = st.session_state.insights
        
        if not filtered_insights:
            st.info("No insights match the selected filters.")
        else:
            for i, insight in enumerate(filtered_insights):
                insight_type = insight.get("insight_type", "concern")
                category = insight.get("category", "communication")
                message = insight.get("message", "")
                confidence = insight.get("confidence", 0.0)
                priority = insight.get("priority", "medium")
                
                # Choose CSS class based on type
                css_class = f"insight-{insight_type.replace('_', '-')}"
                if insight_type == "action_required":
                    css_class = "insight-action"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <h4>📌 {category.title()} Insight</h4>
                    <p><strong>Message:</strong> {message}</p>
                    <p><strong>Priority:</strong> {priority.upper()} | <strong>Confidence:</strong> {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show supporting evidence and actions
                with st.expander(f"Details for Insight {i+1}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**Supporting Evidence:**")
                        evidence = insight.get("supporting_evidence", [])
                        if evidence:
                            for item in evidence:
                                st.write(f"• {item}")
                        else:
                            st.write("No supporting evidence provided")
                    
                    with col_b:
                        st.write("**Recommended Actions:**")
                        actions = insight.get("recommended_actions", [])
                        if actions:
                            for action in actions:
                                st.write(f"• {action}")
                        else:
                            st.write("No specific actions recommended")
                    
                    # Show raw insight data for debugging
                    with st.expander("🔍 Raw Insight Data"):
                        st.json(insight)
    else:
        st.info("Generate insights to see them displayed here")

with tab3:
    st.header("📈 Analytics & Trends")
    
    if st.session_state.processing_results:
        # Processing results over time
        df_results = pd.DataFrame([
            {
                "timestamp": result["timestamp"],
                "action": result["result"].get("action", "unknown"),
                "sentiment": result["result"].get("sentiment", "neutral"),
                "concern_level": result["result"].get("concern_level", "medium"),
                "confidence": result["result"].get("confidence", 0.0),
                "should_flag": result["result"].get("should_flag", False),
                "should_respond": result["result"].get("should_respond", False)
            }
            for result in st.session_state.processing_results
        ])
        
        df_results["timestamp"] = pd.to_datetime(df_results["timestamp"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Action Distribution")
            action_counts = df_results["action"].value_counts()
            fig = px.pie(values=action_counts.values, names=action_counts.index, 
                        title="Message Actions Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("😊 Sentiment Analysis")
            sentiment_counts = df_results["sentiment"].value_counts()
            colors = {"positive": "green", "neutral": "yellow", "negative": "red"}
            fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                        title="Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trends over time
        st.subheader("📈 Trends Over Time")
        
        # Confidence trend
        fig = px.line(df_results, x="timestamp", y="confidence", 
                     title="Confidence Score Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Concern level trend
        concern_mapping = {"low": 1, "medium": 2, "high": 3}
        df_results["concern_numeric"] = df_results["concern_level"].map(concern_mapping)
        
        fig = px.line(df_results, x="timestamp", y="concern_numeric",
                     title="Concern Level Over Time")
        fig.update_yaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"])
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Process some messages to see analytics")

with tab4:
    st.header("🔍 Conversation History")
    
    if st.session_state.conversation_history:
        st.subheader("💬 Messages")
        
        for i, msg in enumerate(reversed(st.session_state.conversation_history)):
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            # Style based on sender
            if sender == "client":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 1rem; margin: 0.5rem 0; border-radius: 10px; border-left: 4px solid #2196f3;">
                    <strong>👤 Client</strong> <small>({timestamp})</small><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 1rem; margin: 0.5rem 0; border-radius: 10px; border-left: 4px solid #9c27b0;">
                    <strong>🤖 System</strong> <small>({timestamp})</small><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        
        # Export conversation
        if st.button("📥 Export Conversation"):
            conversation_json = json.dumps(st.session_state.conversation_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=conversation_json,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("No conversation history yet. Start by processing some messages!")
    
    # Processing results history
    if st.session_state.processing_results:
        st.subheader("📋 Processing Results History")
        
        # Create DataFrame for better display
        results_data = []
        for result in st.session_state.processing_results:
            results_data.append({
                "Timestamp": result["timestamp"],
                "Message": result["message"][:50] + "..." if len(result["message"]) > 50 else result["message"],
                "Action": result["result"].get("action", "unknown"),
                "Sentiment": result["result"].get("sentiment", "neutral"),
                "Concern": result["result"].get("concern_level", "medium"),
                "Confidence": f"{result['result'].get('confidence', 0.0):.1%}",
                "Flag": "🚩" if result["result"].get("should_flag") else "✅",
                "Respond": "✅" if result["result"].get("should_respond") else "❌"
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "🤖 **Arviso AI Agent Testing Dashboard** | "
    "Built with Streamlit | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
