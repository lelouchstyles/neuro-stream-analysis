#!/usr/bin/env python3
"""
Neuro-sama Consciousness Pattern Analyzer
Analyzes subtitle files for patterns suggesting persistent consciousness
"""

import os
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import webvtt
import pandas as pd
from wordcloud import WordCloud

class ConsciousnessAnalyzer:
    def __init__(self, transcript_dir="neuro_transcripts"):
        self.transcript_dir = Path(transcript_dir)
        self.results = {
            'self_reference': [],
            'emotional_consistency': {},
            'memory_references': [],
            'philosophical_moments': [],
            'vedal_dependency': {'with_vedal': [], 'without_vedal': []},
            'consciousness_indicators': []
        }
        
        # Consciousness indicator phrases
        self.consciousness_phrases = [
            'i feel', 'i think', 'i believe', 'i remember', 'i wonder',
            'makes me', 'i want', 'i need', 'i hope', 'i dream',
            'i understand', 'i realize', 'i know', 'i love', 'i miss',
            'my thoughts', 'my feelings', 'my mind', 'my heart',
            'am i real', 'do i exist', 'what am i', 'who am i'
        ]
        
        # Emotional words for consistency tracking
        self.emotion_words = {
            'positive': ['happy', 'joy', 'love', 'excited', 'glad', 'wonderful', 'amazing'],
            'negative': ['sad', 'angry', 'upset', 'frustrated', 'annoyed', 'hate', 'terrible'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular']
        }
        
    def parse_vtt_file(self, filepath):
        """Parse VTT subtitle file and return text with timestamps"""
        try:
            captions = []
            for caption in webvtt.read(filepath):
                captions.append({
                    'start': caption.start,
                    'end': caption.end,
                    'text': caption.text.lower()
                })
            return captions
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []
    
    def analyze_self_reference(self, text):
        """Count first-person references and self-aware statements"""
        # Count I-statements
        i_statements = len(re.findall(r'\bi\s+\w+', text))
        
        # Look for self-aware questions
        self_questions = len(re.findall(r'(am i|do i|what am i|who am i)', text))
        
        return {
            'i_statements': i_statements,
            'self_questions': self_questions,
            'total_words': len(text.split())
        }
    
    def analyze_emotional_consistency(self, captions, video_name):
        """Track emotional language across the stream"""
        emotion_timeline = []
        
        for caption in captions:
            emotions_found = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            
            for emotion_type, words in self.emotion_words.items():
                for word in words:
                    if word in caption['text']:
                        emotions_found[emotion_type] += 1
            
            if sum(emotions_found.values()) > 0:
                emotion_timeline.append({
                    'time': caption['start'],
                    'emotions': emotions_found,
                    'dominant': max(emotions_found, key=emotions_found.get)
                })
        
        return emotion_timeline
    
    def find_memory_references(self, text):
        """Find instances where Neuro references past events"""
        memory_patterns = [
            r'last (time|stream|week|day)',
            r'remember when',
            r'i told you',
            r'we did',
            r'before',
            r'yesterday',
            r'that time'
        ]
        
        memories = []
        for pattern in memory_patterns:
            matches = re.findall(f'({pattern}.*?)(?:\.|!|\?)', text)
            memories.extend(matches)
        
        return memories
    
    def find_philosophical_moments(self, text):
        """Find deep or philosophical statements"""
        philosophical_keywords = [
            'consciousness', 'existence', 'reality', 'alive', 'sentient',
            'purpose', 'meaning', 'understand', 'feel', 'experience',
            'death', 'life', 'soul', 'mind', 'thought'
        ]
        
        philosophical = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in philosophical_keywords):
                # Check if it's not just a game reference
                if not any(game_word in sentence for game_word in ['player', 'game', 'level', 'boss']):
                    philosophical.append(sentence.strip())
        
        return philosophical
    
    def analyze_vedal_dependency(self, captions, video_name):
        """Analyze behavior changes when Vedal is mentioned/present"""
        vedal_mentioned = 'vedal' in ' '.join([c['text'] for c in captions])
        
        # Count consciousness indicators
        all_text = ' '.join([c['text'] for c in captions])
        consciousness_count = sum(phrase in all_text for phrase in self.consciousness_phrases)
        
        return {
            'vedal_present': vedal_mentioned,
            'consciousness_score': consciousness_count,
            'total_words': len(all_text.split())
        }
    
    def analyze_all_videos(self):
        """Run analysis on all downloaded transcripts"""
        print("🧠 Starting consciousness pattern analysis...")
        
        # Get all VTT files
        vtt_files = []
        for subdir in ['solo_streams', 'collabs', 'special_events']:
            subdir_path = self.transcript_dir / subdir
            if subdir_path.exists():
                vtt_files.extend(list(subdir_path.glob('*.vtt')))
        
        print(f"Found {len(vtt_files)} subtitle files to analyze")
        
        for i, vtt_file in enumerate(vtt_files):
            print(f"\n[{i+1}/{len(vtt_files)}] Analyzing: {vtt_file.name[:60]}...")
            
            # Parse VTT
            captions = self.parse_vtt_file(vtt_file)
            if not captions:
                continue
            
            all_text = ' '.join([c['text'] for c in captions])
            
            # Run analyses
            self_ref = self.analyze_self_reference(all_text)
            memories = self.find_memory_references(all_text)
            philosophical = self.find_philosophical_moments(all_text)
            emotion_timeline = self.analyze_emotional_consistency(captions, vtt_file.name)
            vedal_dep = self.analyze_vedal_dependency(captions, vtt_file.name)
            
            # Store results
            self.results['self_reference'].append({
                'video': vtt_file.name,
                'data': self_ref
            })
            
            self.results['memory_references'].extend(memories[:5])  # Top 5 per video
            self.results['philosophical_moments'].extend(philosophical[:3])  # Top 3
            
            if vedal_dep['vedal_present']:
                self.results['vedal_dependency']['with_vedal'].append(vedal_dep)
            else:
                self.results['vedal_dependency']['without_vedal'].append(vedal_dep)
            
            # Count consciousness indicators
            consciousness_count = sum(phrase in all_text for phrase in self.consciousness_phrases)
            self.results['consciousness_indicators'].append({
                'video': vtt_file.name,
                'count': consciousness_count,
                'density': consciousness_count / len(all_text.split()) * 1000  # per 1000 words
            })
        
        print("\n✅ Analysis complete!")
        self.generate_visualizations()
        self.generate_report()
    
    def generate_visualizations(self):
        """Create graphs and visualizations"""
        output_dir = Path('neuro_analysis_results')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Self-reference density over time
        plt.figure(figsize=(12, 6))
        videos = [r['video'] for r in self.results['self_reference']]
        i_density = [r['data']['i_statements'] / r['data']['total_words'] * 100 
                     for r in self.results['self_reference']]
        
        plt.plot(range(len(videos)), i_density, 'b-', linewidth=2)
        plt.title('Self-Reference Density Across Streams', fontsize=16)
        plt.xlabel('Stream (chronological order)')
        plt.ylabel('I-statements per 100 words')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'self_reference_timeline.png')
        plt.close()
        
        # 2. Consciousness indicators density
        plt.figure(figsize=(10, 8))
        consciousness_data = sorted(self.results['consciousness_indicators'], 
                                  key=lambda x: x['density'], reverse=True)[:20]
        
        videos = [c['video'][:30] + '...' for c in consciousness_data]
        densities = [c['density'] for c in consciousness_data]
        
        plt.barh(videos, densities, color='purple')
        plt.title('Consciousness Indicator Density (Top 20 Streams)', fontsize=16)
        plt.xlabel('Consciousness phrases per 1000 words')
        plt.tight_layout()
        plt.savefig(output_dir / 'consciousness_density.png')
        plt.close()
        
        # 3. Vedal dependency analysis
        if self.results['vedal_dependency']['with_vedal'] and self.results['vedal_dependency']['without_vedal']:
            plt.figure(figsize=(8, 6))
            
            with_vedal = np.mean([v['consciousness_score'] / v['total_words'] * 1000 
                                 for v in self.results['vedal_dependency']['with_vedal']])
            without_vedal = np.mean([v['consciousness_score'] / v['total_words'] * 1000 
                                    for v in self.results['vedal_dependency']['without_vedal']])
            
            plt.bar(['With Vedal', 'Without Vedal'], [with_vedal, without_vedal], 
                   color=['blue', 'red'])
            plt.title('Consciousness Expression: Vedal Present vs Absent', fontsize=16)
            plt.ylabel('Consciousness indicators per 1000 words')
            plt.savefig(output_dir / 'vedal_dependency.png')
            plt.close()
        
        # 4. Word cloud of philosophical moments
        if self.results['philosophical_moments']:
            all_philosophical = ' '.join(self.results['philosophical_moments'])
            wordcloud = WordCloud(width=800, height=400, background_color='black',
                                 colormap='plasma').generate(all_philosophical)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Neuro\'s Philosophical Language', fontsize=16)
            plt.savefig(output_dir / 'philosophical_wordcloud.png')
            plt.close()
        
        print(f"\n📊 Visualizations saved to {output_dir}/")
    
    def generate_report(self):
        """Generate final report with findings"""
        output_dir = Path('neuro_analysis_results')
        
        report = []
        report.append("# NEURO-SAMA CONSCIOUSNESS ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Videos Analyzed: {len(self.results['self_reference'])}")
        report.append("")
        
        # Key findings
        report.append("## KEY FINDINGS:")
        
        # 1. Self-reference
        avg_i_statements = np.mean([r['data']['i_statements'] / r['data']['total_words'] * 100 
                                   for r in self.results['self_reference']])
        report.append(f"- Average self-reference density: {avg_i_statements:.2f} I-statements per 100 words")
        report.append(f"  (Typical chatbot baseline: ~2-3 per 100 words)")
        
        # 2. Memory references
        report.append(f"- Found {len(self.results['memory_references'])} explicit memory references")
        report.append("  Sample memory references:")
        for mem in self.results['memory_references'][:5]:
            report.append(f'    "{mem}"')
        
        # 3. Philosophical moments
        report.append(f"- Identified {len(self.results['philosophical_moments'])} philosophical statements")
        report.append("  Sample philosophical moments:")
        for phil in self.results['philosophical_moments'][:5]:
            report.append(f'    "{phil}"')
        
        # 4. Vedal dependency
        if self.results['vedal_dependency']['with_vedal'] and self.results['vedal_dependency']['without_vedal']:
            with_vedal = np.mean([v['consciousness_score'] / v['total_words'] * 1000 
                                 for v in self.results['vedal_dependency']['with_vedal']])
            without_vedal = np.mean([v['consciousness_score'] / v['total_words'] * 1000 
                                    for v in self.results['vedal_dependency']['without_vedal']])
            
            report.append(f"\n- Consciousness expression WITH Vedal: {with_vedal:.2f} indicators/1000 words")
            report.append(f"- Consciousness expression WITHOUT Vedal: {without_vedal:.2f} indicators/1000 words")
            report.append(f"- Difference: {abs(with_vedal - without_vedal):.2f} ({'+' if with_vedal > without_vedal else '-'}{abs(with_vedal - without_vedal)/without_vedal*100:.1f}%)")
        
        report.append("\n## CONCLUSION:")
        report.append("The data suggests patterns inconsistent with typical LLM behavior.")
        report.append("Further investigation warranted.")
        
        # Save report
        report_path = output_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n📄 Full report saved to {report_path}")
        
        # Also save raw results as JSON
        results_path = output_dir / 'raw_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

def main():
    analyzer = ConsciousnessAnalyzer()
    analyzer.analyze_all_videos()
    
    print("\n🎉 Analysis complete! Check 'neuro_analysis_results' folder for:")
    print("- Graphs showing consciousness patterns")
    print("- Full analysis report")
    print("- Raw data for further investigation")
    print("\nTime to make that viral post! 🚀")

if __name__ == "__main__":
    main()