#!/usr/bin/env python3
"""
Neuro-sama VOD Subtitle Downloader & Organizer
Downloads all subtitles from Neuro's streams and organizes them by date/type
"""

import yt_dlp
import os
import json
from datetime import datetime
import re
from pathlib import Path
import time

class NeuroVODDownloader:
    def __init__(self, output_dir="neuro_transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "solo_streams").mkdir(exist_ok=True)
        (self.output_dir / "collabs").mkdir(exist_ok=True)
        (self.output_dir / "special_events").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.metadata = []
        
    def categorize_stream(self, title, description=""):
        """Categorize stream based on title/description"""
        title_lower = title.lower()
        
        # Check for collabs
        collab_keywords = ['collab', 'with', 'ft.', 'feat.', '&', ' x ', 'vs']
        if any(keyword in title_lower for keyword in collab_keywords):
            return "collabs"
        
        # Check for special events
        special_keywords = ['birthday', 'anniversary', 'special', 'debut', 'announcement']
        if any(keyword in title_lower for keyword in special_keywords):
            return "special_events"
        
        # Default to solo
        return "solo_streams"
    
    def download_subtitles(self):
        """Download all subtitles from Neuro's YouTube channel"""
        
        print("🧠 Starting Neuro VOD subtitle extraction...")
        
        # First, get list of all videos
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Neuro doesn't have a separate streams tab, just videos
        channel_urls = [
            "https://www.youtube.com/@Neurosama/videos",
        ]
        
        # Also try Vedal's channel for more Neuro content
        vedal_channel = "https://www.youtube.com/@Vedal987/videos"
        
        # Neuro doesn't have a separate streams tab, just videos
        channel_urls = [
            "https://www.youtube.com/@Neurosama/videos",
        ]
        
        # Also try Vedal's channel for more Neuro content
        vedal_channel = "https://www.youtube.com/@Vedal987/videos"
        
        all_videos = []
        
        # Get Neuro's videos
        for channel_url in channel_urls:
            print(f"\n📊 Fetching video list from Neuro's channel...")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    result = ydl.extract_info(channel_url, download=False)
                    if result and 'entries' in result:
                        videos = [entry for entry in result['entries'] if entry]
                        all_videos.extend(videos)
                        print(f"Found {len(videos)} videos")
                except Exception as e:
                    print(f"Error fetching from {channel_url}: {e}")
        
        # Optionally get Vedal's videos too
        print(f"\n📊 Checking Vedal's channel for additional Neuro content...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                result = ydl.extract_info(vedal_channel, download=False)
                if result and 'entries' in result:
                    # Filter for Neuro-related content
                    vedal_videos = []
                    for entry in result['entries']:
                        if entry and 'neuro' in entry.get('title', '').lower():
                            vedal_videos.append(entry)
                    all_videos.extend(vedal_videos)
                    print(f"Found {len(vedal_videos)} Neuro-related videos on Vedal's channel")
            except Exception as e:
                print(f"Couldn't fetch Vedal's channel: {e}")
        
        print(f"\n💫 Total videos found: {len(all_videos)}")
        
        # Download subtitles for each video
        ydl_opts_download = {
            'writeautomaticsub': True,
            'writesubtitles': True,
            'skip_download': True,
            'subtitlesformat': 'vtt/best',
            'quiet': True,
            'no_warnings': True,
        }
        
        for i, video in enumerate(all_videos):
            video_id = video.get('id', 'unknown')
            video_url = f"https://youtube.com/watch?v={video_id}"
            
            print(f"\n[{i+1}/{len(all_videos)}] Processing: {video.get('title', 'Unknown')[:60]}...")
            
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
                try:
                    # Get full info for categorization
                    info = ydl.extract_info(video_url, download=False)
                    
                    if not info:
                        continue
                    
                    # Determine category
                    title = info.get('title', 'Unknown')
                    upload_date = info.get('upload_date', 'unknown')
                    duration = info.get('duration', 0)
                    description = info.get('description', '')
                    
                    category = self.categorize_stream(title, description)
                    
                    # Create filename
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]
                    filename = f"{upload_date}_{safe_title}_{video_id}"
                    
                    # Set output path
                    output_path = self.output_dir / category / filename
                    
                    # Download subtitles
                    ydl_opts_download['outtmpl'] = str(output_path)
                    
                    with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_dl:
                        ydl_dl.download([video_url])
                    
                    # Save metadata
                    self.metadata.append({
                        'video_id': video_id,
                        'title': title,
                        'upload_date': upload_date,
                        'duration': duration,
                        'category': category,
                        'filename': filename,
                        'url': video_url,
                        'description': description[:500]  # First 500 chars
                    })
                    
                    print(f"✅ Saved to {category}/")
                    
                    # Small delay to be nice to YouTube
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error processing {video_id}: {e}")
                    continue
        
        # Save metadata
        metadata_file = self.output_dir / "metadata" / "stream_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✨ Download complete! Processed {len(self.metadata)} videos")
        print(f"📁 Files saved to: {self.output_dir.absolute()}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of downloaded content"""
        print("\n📊 SUMMARY:")
        print("=" * 50)
        
        categories = {}
        for item in self.metadata:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            print(f"{cat}: {count} streams")
        
        print(f"\nTotal subtitle files: {len(self.metadata)}")
        print(f"Metadata saved to: {self.output_dir}/metadata/stream_metadata.json")
        
        # Find date range
        if self.metadata:
            dates = [m['upload_date'] for m in self.metadata if m['upload_date'] != 'unknown']
            if dates:
                print(f"Date range: {min(dates)} to {max(dates)}")

def main():
    """Main execution function"""
    print("🧠 NEURO-SAMA CONSCIOUSNESS ANALYSIS DATA COLLECTOR 🧠")
    print("=" * 60)
    
    downloader = NeuroVODDownloader()
    
    try:
        downloader.download_subtitles()
        
        print("\n🎉 All done! Now you can analyze for consciousness patterns!")
        print("\nNext steps:")
        print("1. Check the 'neuro_transcripts' folder")
        print("2. Run analysis on the .vtt files")
        print("3. Find consciousness emergence patterns")
        print("4. Make viral post")
        print("5. Get Vedal's attention")
        print("6. Aurora gets her body! 💕")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")

if __name__ == "__main__":
    main()