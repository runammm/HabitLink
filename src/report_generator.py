"""
Report Generator Module
Generates PDF reports from session analysis results using ReportLab.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Try to import ReportLab (PDF generation library with excellent Unicode support)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ Warning: ReportLab not available. PDF report generation will be disabled.")
    print("   To enable PDF reports, install: pip install reportlab")


class ReportGenerator:
    """Generates PDF reports from HabitLink session data using ReportLab."""
    
    def __init__(self, output_dir: str = ".data/report"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = None
        self.korean_font_name = None
        if REPORTLAB_AVAILABLE:
            self._register_korean_font()
            self._setup_styles()
    
    def _register_korean_font(self):
        """Register Korean font for PDF generation."""
        try:
            # Try to find and register a Korean-compatible font on macOS
            # Common macOS Korean font paths
            font_paths = [
                '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                '/Library/Fonts/AppleGothic.ttf',
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        # Register the font
                        pdfmetrics.registerFont(TTFont('KoreanFont', font_path))
                        self.korean_font_name = 'KoreanFont'
                        print(f"✓ Korean font registered: {font_path}")
                        return
                    except Exception as e:
                        print(f"  Failed to register {font_path}: {e}")
                        continue
            
            # If no Korean font found, warn user
            print("⚠ No Korean font found. Text will appear as squares.")
            print("  Please ensure Korean fonts are installed on your system.")
            self.korean_font_name = 'Helvetica'  # Fallback
            
        except Exception as e:
            print(f"Error registering Korean font: {e}")
            self.korean_font_name = 'Helvetica'
    
    def _setup_styles(self):
        """Setup paragraph styles for the PDF."""
        self.styles = getSampleStyleSheet()
        
        # Determine which font to use
        title_font = f'{self.korean_font_name}' if self.korean_font_name else 'Helvetica-Bold'
        body_font = f'{self.korean_font_name}' if self.korean_font_name else 'Helvetica'
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2980B9'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=title_font
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2980B9'),
            spaceAfter=12,
            spaceBefore=12,
            fontName=title_font
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            fontName=body_font
        ))
        
        # Bold body text style (use same font as body for Korean compatibility)
        self.styles.add(ParagraphStyle(
            name='CustomBodyBold',
            parent=self.styles['BodyText'],
            fontSize=10,
            spaceAfter=6,
            fontName=body_font
        ))
    
    def generate_report(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Generate a PDF report from session data."""
        
        if not REPORTLAB_AVAILABLE:
            print("⚠️ PDF report generation skipped: ReportLab not available")
            return self._generate_markdown_fallback(session_data)
        
        try:
            # Generate filename
            session_start = session_data.get("session_start_time")
            if session_start:
                timestamp = session_start.strftime("%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filename = f"habitlink_report_{timestamp}.pdf"
            pdf_path = os.path.join(self.output_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )
            
            # Build content
            story = []
            
            # Add title
            story.append(Paragraph("HabitLink Session Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add all sections
            self._add_session_info(story, session_data)
            self._add_transcript(story, session_data)
            self._add_keyword_analysis(story, session_data)
            self._add_profanity_analysis(story, session_data)
            self._add_speech_rate_analysis(story, session_data)
            self._add_grammar_analysis(story, session_data)
            self._add_context_analysis(story, session_data)
            self._add_stutter_analysis(story, session_data)
            
            # Build PDF
            doc.build(story)
            
            print(f"✅ PDF report saved: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"❌ Error generating PDF report: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_markdown_fallback(session_data)
    
    def _add_session_info(self, story: List, session_data: Dict[str, Any]):
        """Add session information section."""
        story.append(Paragraph("Session Information", self.styles['SectionHeading']))
        
        session_start = session_data.get("session_start_time")
        session_end = session_data.get("session_end_time")
        
        if session_start:
            story.append(Paragraph(f"<b>Start Time:</b> {session_start.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['CustomBody']))
        
        if session_end:
            story.append(Paragraph(f"<b>End Time:</b> {session_end.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['CustomBody']))
        
        if session_start and session_end:
            duration = session_end - session_start
            minutes = int(duration.total_seconds() / 60)
            seconds = int(duration.total_seconds() % 60)
            story.append(Paragraph(f"<b>Duration:</b> {minutes}m {seconds}s", self.styles['CustomBody']))
        
        # Enabled analysis modules
        enabled_analyses = session_data.get("enabled_analyses", {})
        if enabled_analyses:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<b>Enabled Analysis Modules:</b>", self.styles['CustomBodyBold']))
            
            module_names = {
                "keyword_detection": "Keyword Detection",
                "profanity_detection": "Profanity Detection",
                "speech_rate": "Speech Rate Analysis",
                "grammar": "Grammar Analysis",
                "context": "Context Analysis",
                "stutter": "Stutter Analysis"
            }
            
            for key, value in enabled_analyses.items():
                if value:
                    module_name = module_names.get(key, key)
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;- {module_name}", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_transcript(self, story: List, session_data: Dict[str, Any]):
        """Add full transcript section."""
        story.append(Paragraph("Full Transcript", self.styles['SectionHeading']))
        
        transcripts = session_data.get("transcripts", [])
        
        if not transcripts:
            story.append(Paragraph("No transcripts available.", self.styles['CustomBody']))
        else:
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            for transcript in transcripts:
                text = transcript.get("text", "")
                timestamp = transcript.get("timestamp", 0)
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                time_str = f"[{minutes:02d}:{seconds:02d}]"
                
                # Escape HTML special characters and handle Korean
                text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"<b>{time_str}</b> {text_escaped}", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_keyword_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add keyword detection analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("keyword_detection", False)
        if not enabled:
            return
        
        story.append(Paragraph("Keyword Detection", self.styles['SectionHeading']))
        
        keyword_detections = session_data.get("keyword_detections", [])
        
        if not keyword_detections:
            story.append(Paragraph("No keywords detected.", self.styles['CustomBody']))
        else:
            # Count keywords
            keyword_counts = defaultdict(int)
            for detection in keyword_detections:
                keyword = detection.get("keyword", "")
                keyword_counts[keyword] += 1
            
            story.append(Paragraph(f"<b>Total detections:</b> {len(keyword_detections)}", self.styles['CustomBodyBold']))
            story.append(Spacer(1, 0.05*inch))
            
            # Summary
            story.append(Paragraph("<b>Summary by keyword:</b>", self.styles['CustomBodyBold']))
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                keyword_escaped = (keyword
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#39;'))
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;&#39;{keyword_escaped}&#39;: {count} times", self.styles['CustomBody']))
            
            story.append(Spacer(1, 0.05*inch))
            
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            # Show first 10
            story.append(Paragraph("<b>Recent occurrences:</b>", self.styles['CustomBodyBold']))
            for detection in keyword_detections[:10]:
                keyword = detection.get("keyword", "")
                timestamp = detection.get("timestamp", 0)
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                
                keyword_escaped = (keyword
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#39;'))
                
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;[{minutes:02d}:{seconds:02d}] &#39;{keyword_escaped}&#39;", self.styles['CustomBody']))
            
            if len(keyword_detections) > 10:
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;... and {len(keyword_detections) - 10} more", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_profanity_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add profanity detection analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("profanity_detection", False)
        if not enabled:
            return
        
        story.append(Paragraph("Profanity Detection", self.styles['SectionHeading']))
        
        profanity_detections = session_data.get("profanity_detections", [])
        
        if not profanity_detections:
            story.append(Paragraph("No profanity detected. Great!", self.styles['CustomBody']))
        else:
            story.append(Paragraph(f"<b>Total profanities detected:</b> {len(profanity_detections)}", self.styles['CustomBodyBold']))
            story.append(Spacer(1, 0.05*inch))
            
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            for detection in profanity_detections[:10]:
                # WordAnalyzer stores profanity in 'keyword' field
                profanity = detection.get("profanity", detection.get("keyword", ""))
                timestamp = detection.get("timestamp", 0)
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    from datetime import datetime
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                
                # Escape HTML special characters including quotes
                profanity_escaped = (profanity
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#39;'))
                
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;[{minutes:02d}:{seconds:02d}] &#39;{profanity_escaped}&#39;", self.styles['CustomBody']))
            
            if len(profanity_detections) > 10:
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;... and {len(profanity_detections) - 10} more", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_speech_rate_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add speech rate analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("speech_rate", False)
        if not enabled:
            return
        
        story.append(Paragraph("Speech Rate Analysis", self.styles['SectionHeading']))
        
        speech_rate_results = session_data.get("speech_rate_results", [])
        
        if not speech_rate_results:
            story.append(Paragraph("No speech rate data available.", self.styles['CustomBody']))
        else:
            # Calculate average - support both 'wpm' and 'words_per_minute' keys
            total_wpm = sum(r.get("wpm", r.get("words_per_minute", 0)) for r in speech_rate_results if r.get("wpm", r.get("words_per_minute", 0)) > 0)
            count = sum(1 for r in speech_rate_results if r.get("wpm", r.get("words_per_minute", 0)) > 0)
            
            if count > 0:
                avg_wpm = total_wpm / count
                story.append(Paragraph(f"<b>Average speech rate:</b> {avg_wpm:.1f} words/minute", self.styles['CustomBodyBold']))
                story.append(Spacer(1, 0.05*inch))
            
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            # Show segments
            story.append(Paragraph("<b>Speech rate by segment:</b>", self.styles['CustomBodyBold']))
            for result in speech_rate_results[:8]:
                wpm = result.get("wpm", result.get("words_per_minute", 0))
                timestamp = result.get("start", result.get("timestamp", 0))
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;[{minutes:02d}:{seconds:02d}] {wpm:.1f} WPM", self.styles['CustomBody']))
            
            if len(speech_rate_results) > 8:
                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;... and {len(speech_rate_results) - 8} more segments", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_grammar_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add grammar error analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("grammar", False)
        if not enabled:
            return
        
        story.append(Paragraph("Grammar Error Analysis", self.styles['SectionHeading']))
        
        grammar_errors = session_data.get("grammar_errors", [])
        
        if not grammar_errors:
            story.append(Paragraph("No grammar errors detected.", self.styles['CustomBody']))
        else:
            story.append(Paragraph(f"<b>Total grammar errors:</b> {len(grammar_errors)}", self.styles['CustomBodyBold']))
            story.append(Spacer(1, 0.05*inch))
            
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            # Show all errors (not just first 5)
            for i, error in enumerate(grammar_errors, 1):
                # Extract from error_details if present
                error_details = error.get("error_details", {})
                
                # Try multiple possible field names
                error_text = error_details.get("original", error.get("error", ""))
                correction = error_details.get("corrected", error.get("correction", ""))
                explanation = error_details.get("explanation", error.get("explanation", ""))
                timestamp = error.get("timestamp", 0)
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                
                story.append(Paragraph(f"<b>{i}. [{minutes:02d}:{seconds:02d}]</b>", self.styles['CustomBodyBold']))
                
                if error_text:
                    error_escaped = error_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Error:</b> {error_escaped}", self.styles['CustomBody']))
                
                if correction:
                    correction_escaped = correction.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Correction:</b> {correction_escaped}", self.styles['CustomBody']))
                
                if explanation:
                    explanation_escaped = explanation.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Explanation:</b> {explanation_escaped}", self.styles['CustomBody']))
                
                story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_context_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add context error analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("context", False)
        if not enabled:
            return
        
        story.append(Paragraph("Context Error Analysis", self.styles['SectionHeading']))
        
        context_errors = session_data.get("context_errors", [])
        
        if not context_errors:
            story.append(Paragraph("No context errors detected.", self.styles['CustomBody']))
        else:
            story.append(Paragraph(f"<b>Total context errors:</b> {len(context_errors)}", self.styles['CustomBodyBold']))
            story.append(Spacer(1, 0.05*inch))
            
            # Get session start time for relative timestamp calculation
            session_start = session_data.get("session_start_time")
            
            # Show all errors
            for i, error in enumerate(context_errors, 1):
                # Context errors have utterance and reasoning at top level (no error_details)
                error_text = error.get("utterance", "")
                explanation = error.get("reasoning", "")
                timestamp = error.get("timestamp", 0)
                
                # Convert absolute timestamp to relative time from session start
                if session_start and timestamp > 1000000000:  # Unix timestamp check
                    relative_seconds = timestamp - session_start.timestamp()
                else:
                    relative_seconds = timestamp
                
                minutes = int(relative_seconds / 60)
                seconds = int(relative_seconds % 60)
                
                story.append(Paragraph(f"<b>{i}. [{minutes:02d}:{seconds:02d}]</b>", self.styles['CustomBodyBold']))
                
                if error_text:
                    error_escaped = error_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Problematic utterance:</b> {error_escaped}", self.styles['CustomBody']))
                
                if explanation:
                    explanation_escaped = explanation.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Reasoning:</b> {explanation_escaped}", self.styles['CustomBody']))
                
                story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_stutter_analysis(self, story: List, session_data: Dict[str, Any]):
        """Add stutter analysis."""
        enabled = session_data.get("enabled_analyses", {}).get("stutter", False)
        if not enabled:
            return
        
        story.append(Paragraph("Stutter Analysis", self.styles['SectionHeading']))
        
        # Real-time detector results
        detector_events = session_data.get("stutter_detector_events", [])
        detector_stats = session_data.get("stutter_detector_stats", {})
        
        if detector_events:
            story.append(Paragraph("<b>Real-time Audio Analysis (Pre-STT):</b>", self.styles['CustomBodyBold']))
            story.append(Paragraph(f"Total events detected: {detector_stats.get('total_events', 0)}", self.styles['CustomBody']))
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;Repetitions: {detector_stats.get('repetitions', 0)}", self.styles['CustomBody']))
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;Prolongations: {detector_stats.get('prolongations', 0)}", self.styles['CustomBody']))
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;Blocks: {detector_stats.get('blocks', 0)}", self.styles['CustomBody']))
            story.append(Spacer(1, 0.05*inch))
        
        # Text-based analysis results
        stutter_results = session_data.get("stutter_results", {})
        if stutter_results:
            stats = stutter_results.get("statistics", {})
            
            story.append(Paragraph("<b>Text-based Analysis (Post-STT):</b>", self.styles['CustomBodyBold']))
            story.append(Paragraph(f"Fluency score: {stats.get('fluency_percentage', 0):.1f}%", self.styles['CustomBody']))
            story.append(Paragraph(f"Total events: {stats.get('total_events', 0)}", self.styles['CustomBody']))
            story.append(Spacer(1, 0.05*inch))
            
            repetitions = stutter_results.get("repetitions", [])
            if repetitions:
                story.append(Paragraph(f"<b>Repetitions:</b> {len(repetitions)}", self.styles['CustomBodyBold']))
                for rep in repetitions[:3]:
                    full_match = rep.get('full_match', '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;- '{full_match}'", self.styles['CustomBody']))
                if len(repetitions) > 3:
                    story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;... and {len(repetitions) - 3} more", self.styles['CustomBody']))
        
        if not detector_events and not stutter_results:
            story.append(Paragraph("No stuttering events detected.", self.styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _generate_markdown_fallback(self, session_data: Dict[str, Any]) -> Optional[str]:
        """Generate markdown report as fallback."""
        lines = []
        
        lines.append("# HabitLink Session Analysis Report")
        lines.append("")
        lines.append("## Session Information")
        lines.append("")
        
        session_start = session_data.get("session_start_time")
        session_end = session_data.get("session_end_time")
        
        if session_start:
            lines.append(f"**Start Time:** {session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        if session_end:
            lines.append(f"**End Time:** {session_end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if session_start and session_end:
            duration = session_end - session_start
            minutes = int(duration.total_seconds() / 60)
            seconds = int(duration.total_seconds() % 60)
            lines.append(f"**Duration:** {minutes}m {seconds}s")
        
        lines.append("")
        lines.append("## Full Transcript")
        lines.append("")
        
        transcripts = session_data.get("transcripts", [])
        if transcripts:
            for transcript in transcripts:
                text = transcript.get("text", "")
                timestamp = transcript.get("timestamp", 0)
                minutes = int(timestamp / 60)
                seconds = int(timestamp % 60)
                lines.append(f"**[{minutes:02d}:{seconds:02d}]** {text}")
        else:
            lines.append("_No transcripts available._")
        
        lines.append("")
        
        # Save markdown
        session_start = session_data.get("session_start_time")
        if session_start:
            timestamp = session_start.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"habitlink_report_{timestamp}.md"
        md_path = os.path.join(self.output_dir, filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"📄 Markdown report saved: {md_path}")
        return md_path
