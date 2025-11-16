import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from collections import Counter


import plotly.graph_objects as go
import plotly.express as px
# Page configuration
st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #AADBF1;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .header-style {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Helper Functions
def analyze_sentiment_simple(text):
    """Simple sentiment analysis based on keywords"""
    positive_words = ['good', 'great', 'excellent', 'confident', 'skilled', 'experienced', 
                     'proficient', 'successful', 'achieved', 'strong', 'passionate', 'love', 
                     'enjoy', 'excited', 'innovative', 'efficient', 'effective']
    negative_words = ['difficult', 'challenging', 'struggle', 'unsure', 'nervous', 'worried',
                     'problem', 'issue', 'failed', 'weak', 'lacking', 'unfortunately']
    neutral_words = ['okay', 'fine', 'average', 'normal', 'standard']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    neu_count = sum(1 for word in neutral_words if word in text_lower)
    
    total = pos_count + neg_count + neu_count
    if total == 0:
        return 'Neutral', 0.5
    
    pos_score = pos_count / total
    neg_score = neg_count / total
    
    if pos_score > neg_score:
        return 'Positive', 0.5 + (pos_score * 0.5)
    elif neg_score > pos_score:
        return 'Negative', 0.5 - (neg_score * 0.5)
    else:
        return 'Neutral', 0.5

def detect_filler_words(text):
    """Detect filler words in speech"""
    filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 
                   'just', 'sort of', 'kind of', 'i mean', 'so']
    text_lower = text.lower()
    filler_count = {}
    total_fillers = 0
    
    for filler in filler_words:
        count = len(re.findall(r'\b' + filler + r'\b', text_lower))
        if count > 0:
            filler_count[filler] = count
            total_fillers += count
    
    return filler_count, total_fillers

def calculate_confidence_score(text):
    """Calculate confidence based on linguistic patterns"""
    confident_phrases = ['i can', 'i will', 'i have', 'i am confident', 'definitely', 
                        'certainly', 'absolutely', 'i believe', 'i think', 'my experience']
    uncertain_phrases = ['maybe', 'perhaps', 'i guess', 'not sure', 'i dont know', 
                        'possibly', 'might', 'could be']
    
    text_lower = text.lower()
    confident_count = sum(1 for phrase in confident_phrases if phrase in text_lower)
    uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in text_lower)
    
    word_count = len(text.split())
    
    # Confidence score (0-100)
    if word_count > 0:
        confidence = max(0, min(100, 50 + (confident_count * 10) - (uncertain_count * 10)))
    else:
        confidence = 50
    
    return confidence

def extract_keywords(text, top_n=10):
    """Extract key topics from text"""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'can', 'could', 'should', 'may', 'might', 'i', 'you', 'he', 'she',
                 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered_words = [w for w in words if w not in stop_words]
    word_freq = Counter(filtered_words)
    
    return word_freq.most_common(top_n)

def analyze_response_quality(text):
    """Analyze the quality of responses"""
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text))
    
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Quality metrics
    has_examples = bool(re.search(r'\bfor example\b|\bfor instance\b|\bsuch as\b', text.lower()))
    has_structure = bool(re.search(r'\bfirst\b|\bsecond\b|\bfinally\b|\bin conclusion\b', text.lower()))
    
    quality_score = 0
    if 50 < word_count < 300:  # Appropriate length
        quality_score += 30
    if 10 < avg_sentence_length < 25:  # Good sentence length
        quality_score += 25
    if has_examples:
        quality_score += 25
    if has_structure:
        quality_score += 20
    
    return min(100, quality_score)

def parse_transcript(transcript):
    """Parse transcript into speaker segments"""
    lines = transcript.strip().split('\n')
    segments = []
    
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            text = parts[1].strip()
            segments.append({'speaker': speaker, 'text': text})
    
    return segments

def analyze_interview(transcript, domain, round_type, feedback_tone):
    """Main analysis function"""
    segments = parse_transcript(transcript)
    
    if not segments:
        return None
    
    # Aggregate analysis
    speaker_stats = {}
    sentiment_timeline = []
    confidence_timeline = []
    all_keywords = []
    
    for idx, segment in enumerate(segments):
        speaker = segment['speaker']
        text = segment['text']
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'total_words': 0,
                'segments': 0,
                'sentiments': [],
                'confidence_scores': [],
                'filler_words': {},
                'quality_scores': []
            }
        
        # Analyze each segment
        word_count = len(text.split())
        sentiment, sentiment_score = analyze_sentiment_simple(text)
        confidence = calculate_confidence_score(text)
        fillers, filler_count = detect_filler_words(text)
        quality = analyze_response_quality(text)
        keywords = extract_keywords(text)
        
        # Update speaker stats
        speaker_stats[speaker]['total_words'] += word_count
        speaker_stats[speaker]['segments'] += 1
        speaker_stats[speaker]['sentiments'].append(sentiment_score)
        speaker_stats[speaker]['confidence_scores'].append(confidence)
        speaker_stats[speaker]['quality_scores'].append(quality)
        
        for filler, count in fillers.items():
            speaker_stats[speaker]['filler_words'][filler] = \
                speaker_stats[speaker]['filler_words'].get(filler, 0) + count
        
        # Timeline data
        if 'Candidate' in speaker or 'Interviewee' in speaker:
            sentiment_timeline.append({
                'segment': idx + 1,
                'sentiment': sentiment_score,
                'confidence': confidence
            })
        
        all_keywords.extend([kw[0] for kw in keywords])
    
    # Calculate overall metrics
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats['avg_sentiment'] = np.mean(stats['sentiments'])
        stats['avg_confidence'] = np.mean(stats['confidence_scores'])
        stats['avg_quality'] = np.mean(stats['quality_scores'])
        stats['total_fillers'] = sum(stats['filler_words'].values())
        stats['filler_rate'] = stats['total_fillers'] / max(stats['total_words'], 1) * 100
    
    # Extract top keywords
    top_keywords = Counter(all_keywords).most_common(15)
    
    # Generate insights based on feedback tone
    insights = generate_insights(speaker_stats, domain, round_type, feedback_tone)
    
    return {
        'speaker_stats': speaker_stats,
        'sentiment_timeline': sentiment_timeline,
        'top_keywords': top_keywords,
        'insights': insights,
        'segments': segments
    }

def generate_insights(speaker_stats, domain, round_type, feedback_tone):
    """Generate contextual insights and recommendations"""
    insights = {
        'strengths': [],
        'improvements': [],
        'recommendations': []
    }
    
    for speaker, stats in speaker_stats.items():
        if 'Candidate' in speaker or 'Interviewee' in speaker:
            # Analyze strengths
            if stats['avg_confidence'] > 70:
                insights['strengths'].append(f"Strong confidence level ({stats['avg_confidence']:.1f}/100)")
            if stats['avg_sentiment'] > 0.6:
                insights['strengths'].append("Positive and enthusiastic communication")
            if stats['avg_quality'] > 60:
                insights['strengths'].append("Well-structured and clear responses")
            if stats['filler_rate'] < 3:
                insights['strengths'].append("Minimal use of filler words")
            
            # Analyze areas for improvement
            if stats['avg_confidence'] < 50:
                insights['improvements'].append("Build more confidence in responses - use assertive language")
            if stats['filler_rate'] > 5:
                insights['improvements'].append(f"Reduce filler words (currently {stats['filler_rate']:.1f}%) - practice pausing instead")
            if stats['avg_quality'] < 40:
                insights['improvements'].append("Provide more structured answers with examples")
            if stats['avg_sentiment'] < 0.4:
                insights['improvements'].append("Show more enthusiasm and positive attitude")
            
            # Generate recommendations based on domain
            if domain == 'technical':
                insights['recommendations'].extend([
                    "Use the STAR method (Situation, Task, Action, Result) for technical questions",
                    "Include specific technologies and metrics in your responses",
                    "Prepare examples of problem-solving and debugging experiences"
                ])
            elif domain == 'managerial':
                insights['recommendations'].extend([
                    "Demonstrate leadership experiences with concrete examples",
                    "Highlight team collaboration and conflict resolution skills",
                    "Quantify achievements and impact on team/organization"
                ])
            elif domain == 'hr':
                insights['recommendations'].extend([
                    "Show cultural fit and alignment with company values",
                    "Discuss work-life balance and career growth aspirations",
                    "Be authentic and show emotional intelligence"
                ])
    
    # Adjust tone based on feedback preference
    if feedback_tone == 'encouraging':
        insights['tone_message'] = "You're on the right track! With these improvements, you'll excel in your interviews."
    elif feedback_tone == 'critical':
        insights['tone_message'] = "Focus on these critical areas to significantly improve your interview performance."
    else:  # professional
        insights['tone_message'] = "Consider these insights to enhance your interview effectiveness."
    
    return insights

# ===== STREAMLIT UI =====

# Header
st.markdown("""
    <div class="header-style">
        <h1>🎯 AI Interview Analyzer</h1>
        <p>Real-Time Conversation Intelligence & Performance Insights</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    input_type = st.radio(
        "Input Type",
        ["Text Transcript", "Audio File (Future)"],
        help="Choose your input format"
    )
    
    st.markdown("---")
    
    domain = st.selectbox(
        "Interview Domain",
        ["Technical", "Managerial", "HR", "Group Discussion"],
        help="Select the interview type for context-aware analysis"
    )
    
    round_type = st.selectbox(
        "Round Type",
        ["One-on-One", "Panel Interview", "Group Discussion", "Behavioral Round"],
        help="Specify the interview format"
    )
    
    feedback_tone = st.selectbox(
        "Feedback Tone",
        ["Professional", "Encouraging", "Critical"],
        help="Choose how you'd like to receive feedback"
    )
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 How to Use
    1. Select input type and configuration
    2. Paste your interview transcript
    3. Click 'Analyze Interview'
    4. Review detailed insights and visualizations
    5. Download comprehensive report
    """)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Interview Data")
    
    if input_type == "Audio File (Future)":
        st.info("🎤 Audio processing will be implemented using OpenAI Whisper or AssemblyAI")
        uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'm4a'])
        if uploaded_file:
            st.audio(uploaded_file)
            st.warning("Demo: Please use Text Transcript mode for current functionality")
    
    # Sample transcript button
    if st.button("📋 Load Sample Interview"):
        sample = """Interviewer: Tell me about yourself and your experience with software development.
Candidate: Um, well, I have been working in software development for about 5 years now. I started with Java and then moved to Python. I have worked on several projects involving machine learning and data analysis. I am passionate about creating efficient solutions and I really enjoy problem-solving.
Interviewer: Can you describe a challenging project you worked on?
Candidate: Sure! I worked on a project where we had to optimize a recommendation system. The challenge was that the system was taking too long to process user data. I analyzed the code, identified bottlenecks, and implemented caching mechanisms. As a result, we reduced processing time by 60%. It was definitely a learning experience.
Interviewer: How do you handle working under pressure or tight deadlines?
Candidate: I think... um... well, I try to stay organized. I break down tasks into smaller pieces and prioritize them. I also communicate with my team regularly to make sure everyone is on the same page. Sometimes it gets stressful, but I believe in staying calm and focused.
Interviewer: What are your salary expectations?
Candidate: I am looking for something in the range of 80 to 90 thousand dollars annually, but I am open to discussion based on the overall compensation package and growth opportunities.
Interviewer: Do you have any questions for us?
Candidate: Yes! I would like to know more about the team structure and the technologies you are currently using. Also, what does the career growth path look like for this position?"""
        st.session_state.sample_loaded = sample
    
    transcript_input = st.text_area(
        "Paste Interview Transcript",
        value=st.session_state.get('sample_loaded', ''),
        height=300,
        placeholder="Format:\nSpeaker1: Text here\nSpeaker2: Text here\n\nExample:\nInterviewer: Tell me about yourself.\nCandidate: I have 5 years of experience...",
        help="Each line should start with 'Speaker:' followed by their dialogue"
    )

with col2:
    st.subheader("🎯 Quick Stats")
    if transcript_input:
        word_count = len(transcript_input.split())
        line_count = len([l for l in transcript_input.split('\n') if l.strip()])
        
        st.metric("Total Words", f"{word_count:,}")
        st.metric("Total Exchanges", line_count)
        st.metric("Avg Words/Exchange", f"{word_count//max(line_count,1)}")
    else:
        st.info("Enter transcript to see statistics")

# Analysis Button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    analyze_button = st.button("🔍 Analyze Interview", type="primary", use_container_width=True)

# Perform Analysis
if analyze_button and transcript_input:
    with st.spinner("🤖 Analyzing interview with AI... This may take a moment..."):
        import time
        time.sleep(2)  # Simulate processing
        
        analysis_data = analyze_interview(
            transcript_input,
            domain.lower(),
            round_type.lower(),
            feedback_tone.lower()
        )
        
        if analysis_data:
            st.session_state.analysis_complete = True
            st.session_state.analysis_data = analysis_data
            st.success("✅ Analysis Complete!")
        else:
            st.error("❌ Could not parse transcript. Please check the format.")

# Display Results
if st.session_state.analysis_complete and st.session_state.analysis_data:
    data = st.session_state.analysis_data
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Key Metrics Row
    st.subheader("🎯 Overall Performance Metrics")
    metrics_cols = st.columns(4)
    
    candidate_stats = None
    for speaker, stats in data['speaker_stats'].items():
        if 'Candidate' in speaker or 'Interviewee' in speaker:
            candidate_stats = stats
            break
    
    if candidate_stats:
        with metrics_cols[0]:
            st.metric(
                "Confidence Score",
                f"{candidate_stats['avg_confidence']:.1f}/100",
                delta=f"{candidate_stats['avg_confidence']-70:.1f}",
                delta_color="normal"
            )
        with metrics_cols[1]:
            sentiment_pct = candidate_stats['avg_sentiment'] * 100
            st.metric(
                "Sentiment Score",
                f"{sentiment_pct:.1f}%",
                delta=f"{sentiment_pct-50:.1f}",
                delta_color="normal"
            )
        with metrics_cols[2]:
            st.metric(
                "Response Quality",
                f"{candidate_stats['avg_quality']:.1f}/100",
                delta=f"{candidate_stats['avg_quality']-60:.1f}",
                delta_color="normal"
            )
        with metrics_cols[3]:
            st.metric(
                "Filler Word Rate",
                f"{candidate_stats['filler_rate']:.1f}%",
                delta=f"{3-candidate_stats['filler_rate']:.1f}",
                delta_color="inverse"
            )
    
    # Visualization Section
    st.markdown("---")
    st.subheader("📈 Performance Visualizations")
    
    # Two column layout for charts
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Sentiment & Confidence Timeline
        if data['sentiment_timeline']:
            timeline_df = pd.DataFrame(data['sentiment_timeline'])
            
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['segment'],
                y=timeline_df['sentiment'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['segment'],
                y=timeline_df['confidence']/100,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=8)
            ))
            
            fig_timeline.update_layout(
                title="Sentiment & Confidence Over Time",
                xaxis_title="Response Segment",
                yaxis_title="Score (0-1)",
                hovermode='x unified',
                height=400,
                showlegend=True,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with viz_col2:
        # Speaker Performance Radar
        if candidate_stats:
            categories = ['Confidence', 'Sentiment', 'Quality', 'Clarity']
            values = [
                candidate_stats['avg_confidence'],
                candidate_stats['avg_sentiment'] * 100,
                candidate_stats['avg_quality'],
                max(0, 100 - candidate_stats['filler_rate'] * 10)
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.5)',
                line=dict(color='#667eea', width=2),
                name='Performance'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=False,
                title="Performance Radar",
                height=400
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Second row of visualizations
    viz_col3, viz_col4 = st.columns(2)
    
    with viz_col3:
        # Filler Words Distribution
        if candidate_stats and candidate_stats['filler_words']:
            filler_df = pd.DataFrame([
                {'word': k, 'count': v} 
                for k, v in sorted(candidate_stats['filler_words'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            ])
            
            fig_filler = px.bar(
                filler_df,
                x='count',
                y='word',
                orientation='h',
                title='Top Filler Words Usage',
                color='count',
                color_continuous_scale='Reds',
                labels={'count': 'Frequency', 'word': 'Filler Word'}
            )
            fig_filler.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_filler, use_container_width=True)
        else:
            st.info("No significant filler words detected - Great job!")
    
    with viz_col4:
        # Top Keywords WordCloud Style
        if data['top_keywords']:
            keywords_df = pd.DataFrame(data['top_keywords'], columns=['keyword', 'frequency'])
            
            fig_keywords = px.treemap(
                keywords_df,
                path=['keyword'],
                values='frequency',
                title='Key Topics Discussed',
                color='frequency',
                color_continuous_scale='Viridis'
            )
            fig_keywords.update_layout(height=400)
            st.plotly_chart(fig_keywords, use_container_width=True)
    
    # Speaker Comparison
    st.markdown("---")
    st.subheader("👥 Speaker Analysis")
    
    speaker_comparison = []
    for speaker, stats in data['speaker_stats'].items():
        speaker_comparison.append({
            'Speaker': speaker,
            'Total Words': stats['total_words'],
            'Segments': stats['segments'],
            'Avg Confidence': f"{stats['avg_confidence']:.1f}",
            'Sentiment': f"{stats['avg_sentiment']*100:.1f}%",
            'Quality': f"{stats['avg_quality']:.1f}",
            'Filler Rate': f"{stats['filler_rate']:.2f}%"
        })
    
    speaker_df = pd.DataFrame(speaker_comparison)
    st.dataframe(speaker_df, use_container_width=True, hide_index=True)
    
    # Insights and Recommendations
    st.markdown("---")
    st.subheader("💡 AI-Generated Insights")
    
    insights = data['insights']
    
    col_insights1, col_insights2 = st.columns(2)
    
    with col_insights1:
        st.markdown("### ✅ Strengths")
        if insights['strengths']:
            for strength in insights['strengths']:
                st.markdown(f"- {strength}")
        else:
            st.info("Keep working on building your interview strengths!")
    
    with col_insights2:
        st.markdown("### 🎯 Areas for Improvement")
        if insights['improvements']:
            for improvement in insights['improvements']:
                st.markdown(f"- {improvement}")
        else:
            st.success("Excellent performance across all areas!")
    
    st.markdown("### 📚 Recommendations")
    if insights['recommendations']:
        for idx, rec in enumerate(insights['recommendations'], 1):
            st.markdown(f"{idx}. {rec}")
    
    st.markdown(f"""
        <div class="insight-box">
            <h4>💬 Feedback Message</h4>
            <p>{insights['tone_message']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Transcript with Annotations
    st.markdown("---")
    st.subheader("📝 Annotated Transcript")
    
    with st.expander("View Detailed Transcript Analysis"):
        for idx, segment in enumerate(data['segments'], 1):
            speaker = segment['speaker']
            text = segment['text']
            sentiment, sentiment_score = analyze_sentiment_simple(text)
            confidence = calculate_confidence_score(text)
            
            sentiment_color = 'green' if sentiment == 'Positive' else ('red' if sentiment == 'Negative' else 'gray')
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid {sentiment_color};">
                    <strong>#{idx} - {speaker}</strong><br>
                    <span style="color: #666;">Sentiment: {sentiment} ({sentiment_score:.2f}) | Confidence: {confidence:.0f}/100</span><br>
                    <p style="margin-top: 10px;">{text}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Export Report
    st.markdown("---")
    st.subheader("📥 Export Report")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        # Prepare JSON export
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'round_type': round_type,
            'speaker_stats': {k: {
                'avg_confidence': float(v['avg_confidence']),
                'avg_sentiment': float(v['avg_sentiment']),
                'avg_quality': float(v['avg_quality']),
                'filler_rate': float(v['filler_rate'])
            } for k, v in data['speaker_stats'].items()},
            'insights': insights
        }
        
        st.download_button(
            label="📄 Download JSON Report",
            data=json.dumps(export_data, indent=2),
            file_name=f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col_export2:
        # CSV export of speaker stats
        st.download_button(
            label="📊 Download CSV Report",
            data=speaker_df.to_csv(index=False),
            file_name=f"speaker_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export3:
        st.info("📋 PDF export not  available ")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 Powered by AI Analysis | Built with Streamlit & Plotly</p>
        <p style="font-size: 0.9em;">For best results, ensure clear speaker labels in transcript format</p>
    </div>

    """, unsafe_allow_html=True)



