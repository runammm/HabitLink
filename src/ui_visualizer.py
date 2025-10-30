import pygame
import numpy as np
import math
from queue import Queue, Empty
from collections import deque
import time
from typing import Optional, Dict, Any


class VoiceVisualizer:
    """
    Pygame-based UI for real-time voice visualization.
    Shows a 3D sphere that changes size based on volume,
    with waveform overlay and detection alerts.
    """
    
    # Window settings
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FPS = 60
    
    # Colors
    COLOR_BACKGROUND = (255, 255, 255)  # White
    COLOR_SPHERE = (30, 144, 255)  # Blue (Dodger Blue)
    COLOR_SPHERE_ALERT = (144, 238, 144)  # Light Green
    COLOR_WAVEFORM = (70, 130, 180)  # Steel Blue
    COLOR_TEXT = (50, 50, 50)  # Dark Gray
    COLOR_ALERT = (0, 200, 0)  # Green
    
    # Sphere settings
    SPHERE_BASE_RADIUS = 80
    SPHERE_MIN_RADIUS = 60
    SPHERE_MAX_RADIUS = 150
    
    # Waveform settings
    WAVEFORM_LENGTH = 200  # Number of samples to display
    WAVEFORM_AMPLITUDE = 60  # Max amplitude for display
    
    # Alert settings
    ALERT_DURATION = 2.0  # seconds
    
    def __init__(self, audio_queue: Queue, feedback_queue: Queue):
        """
        Initialize the visualizer.
        
        Args:
            audio_queue: Queue containing audio data for waveform visualization
            feedback_queue: Queue containing detection feedback messages
        """
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("HabitLink - Voice Visualizer")
        
        self.clock = pygame.time.Clock()
        
        # Use fonts that support Korean characters
        # Try to use system fonts that support Korean
        korean_fonts = ['AppleSDGothicNeo', 'AppleGothic', 'NanumGothic', 'Malgun Gothic', 'Gulim']
        font_large = None
        font_small = None
        
        for font_name in korean_fonts:
            try:
                font_large = pygame.font.SysFont(font_name, 24, bold=True)
                font_small = pygame.font.SysFont(font_name, 18)
                # Test if Korean works
                test_surface = font_large.render('테스트', True, (0, 0, 0))
                break
            except:
                continue
        
        # Fallback to default font if Korean fonts not found
        if font_large is None:
            font_large = pygame.font.Font(None, 28)
            font_small = pygame.font.Font(None, 20)
        
        self.font_large = font_large
        self.font_small = font_small
        
        self.audio_queue = audio_queue
        self.feedback_queue = feedback_queue
        
        # State
        self.is_running = True
        self.current_volume = 0.0
        self.target_volume = 0.0
        self.current_radius = self.SPHERE_BASE_RADIUS
        self.waveform_data = deque([0] * self.WAVEFORM_LENGTH, maxlen=self.WAVEFORM_LENGTH)
        
        # Sphere color animation
        self.is_alert_mode = False
        self.alert_start_time = 0
        
        # Active alerts
        self.active_alerts = []  # List of (message, start_time) tuples
        
        # Center position
        self.center_x = self.WINDOW_WIDTH // 2
        self.center_y = self.WINDOW_HEIGHT // 2 - 50
    
    def process_audio_data(self):
        """Process audio data from the queue for visualization."""
        try:
            while not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get_nowait()
                
                # Convert bytes to numpy array
                if isinstance(audio_chunk, bytes):
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                else:
                    audio_array = audio_chunk
                
                # Calculate volume (RMS)
                if len(audio_array) > 0:
                    rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
                    # Normalize to 0-1 range (assuming 16-bit audio)
                    volume = min(rms / 10000.0, 1.0)
                    self.target_volume = volume
                    
                    # Add to waveform (downsample if needed)
                    downsample_factor = max(1, len(audio_array) // 100)
                    downsampled = audio_array[::downsample_factor]
                    
                    for sample in downsampled:
                        # Normalize to -1 to 1
                        normalized = sample / 32768.0
                        self.waveform_data.append(normalized)
        
        except Empty:
            pass
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def process_feedback(self):
        """Process feedback messages from the queue."""
        try:
            while not self.feedback_queue.empty():
                feedback = self.feedback_queue.get_nowait()
                
                # Parse feedback message
                if isinstance(feedback, dict):
                    message = feedback.get("message", "")
                    alert_type = feedback.get("type", "info")
                else:
                    message = str(feedback)
                    alert_type = "info"
                
                # Add to active alerts
                self.active_alerts.append({
                    "message": message,
                    "start_time": time.time(),
                    "type": alert_type
                })
                
                # Trigger alert mode
                self.is_alert_mode = True
                self.alert_start_time = time.time()
        
        except Empty:
            pass
        except Exception as e:
            print(f"Error processing feedback: {e}")
    
    def update_state(self):
        """Update visualization state."""
        # Smooth volume transition
        smoothing = 0.2
        self.current_volume += (self.target_volume - self.current_volume) * smoothing
        
        # Update sphere radius based on volume
        volume_factor = self.current_volume
        target_radius = self.SPHERE_BASE_RADIUS + (self.SPHERE_MAX_RADIUS - self.SPHERE_BASE_RADIUS) * volume_factor
        target_radius = max(self.SPHERE_MIN_RADIUS, min(self.SPHERE_MAX_RADIUS, target_radius))
        
        # Smooth radius transition
        self.current_radius += (target_radius - self.current_radius) * 0.15
        
        # Check alert timeout
        current_time = time.time()
        if self.is_alert_mode and (current_time - self.alert_start_time) > self.ALERT_DURATION:
            self.is_alert_mode = False
        
        # Remove old alerts
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - alert["start_time"]) < 5.0  # Keep for 5 seconds
        ]
        
        # Decay target volume
        self.target_volume *= 0.95
    
    def draw_sphere(self):
        """Draw the 3D sphere with gradient effect."""
        # Choose color based on alert mode
        if self.is_alert_mode:
            base_color = self.COLOR_SPHERE_ALERT
        else:
            base_color = self.COLOR_SPHERE
        
        radius = int(self.current_radius)
        
        # Draw multiple circles with varying opacity for 3D effect
        for i in range(radius, 0, -2):
            # Calculate color gradient (darker towards edges)
            factor = i / radius
            color = tuple(int(c * factor + (255 - c) * (1 - factor) * 0.3) for c in base_color)
            
            pygame.draw.circle(
                self.screen,
                color,
                (self.center_x, self.center_y),
                i
            )
        
        # Draw highlight for 3D effect
        highlight_offset_x = -radius // 4
        highlight_offset_y = -radius // 4
        highlight_radius = radius // 3
        
        for i in range(highlight_radius, 0, -1):
            alpha = int(200 * (i / highlight_radius))
            color = (
                min(255, base_color[0] + 50),
                min(255, base_color[1] + 50),
                min(255, base_color[2] + 50)
            )
            pygame.draw.circle(
                self.screen,
                color,
                (self.center_x + highlight_offset_x, self.center_y + highlight_offset_y),
                i
            )
    
    def draw_waveform(self):
        """Draw waveform across the sphere diameter."""
        if len(self.waveform_data) < 2:
            return
        
        radius = int(self.current_radius)
        waveform_width = radius * 2
        
        # Calculate points for waveform
        points = []
        waveform_list = list(self.waveform_data)
        
        # Use the most recent samples
        num_samples = min(len(waveform_list), 100)
        samples = waveform_list[-num_samples:]
        
        for i, sample in enumerate(samples):
            x = self.center_x - radius + (i / (num_samples - 1)) * waveform_width
            y = self.center_y + sample * self.WAVEFORM_AMPLITUDE
            points.append((int(x), int(y)))
        
        # Draw waveform line
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLOR_WAVEFORM, False, points, 2)
        
        # Draw center line
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (self.center_x - radius, self.center_y),
            (self.center_x + radius, self.center_y),
            1
        )
    
    def draw_alerts(self):
        """Draw active alerts in the top-right corner."""
        if not self.active_alerts:
            return
        
        x = self.WINDOW_WIDTH - 20
        y = 20
        
        for alert in self.active_alerts[-3:]:  # Show last 3 alerts
            message = alert["message"]
            
            # Create text surface
            text_surface = self.font_small.render(message, True, self.COLOR_ALERT)
            text_rect = text_surface.get_rect()
            text_rect.topright = (x, y)
            
            # Draw semi-transparent background
            bg_rect = text_rect.inflate(20, 10)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
            bg_surface.set_alpha(200)
            bg_surface.fill((240, 255, 240))
            self.screen.blit(bg_surface, bg_rect.topleft)
            
            # Draw text
            self.screen.blit(text_surface, text_rect)
            
            y += text_rect.height + 15
    
    def draw_status(self):
        """Draw status information at the bottom."""
        status_text = f"Volume: {int(self.current_volume * 100)}%"
        text_surface = self.font_small.render(status_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect()
        text_rect.center = (self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT - 30)
        self.screen.blit(text_surface, text_rect)
    
    def draw_title(self):
        """Draw title at the top."""
        title_text = "HabitLink - 실시간 음성 분석"
        text_surface = self.font_large.render(title_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect()
        text_rect.center = (self.WINDOW_WIDTH // 2, 30)
        self.screen.blit(text_surface, text_rect)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
    
    def run(self):
        """Main visualization loop."""
        print("🎨 UI Visualizer started")
        
        while self.is_running:
            # Handle events
            self.handle_events()
            
            # Process data
            self.process_audio_data()
            self.process_feedback()
            
            # Update state
            self.update_state()
            
            # Clear screen
            self.screen.fill(self.COLOR_BACKGROUND)
            
            # Draw elements
            self.draw_title()
            self.draw_sphere()
            self.draw_waveform()
            self.draw_alerts()
            self.draw_status()
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(self.FPS)
        
        # Cleanup
        pygame.quit()
        print("🎨 UI Visualizer stopped")
    
    def stop(self):
        """Stop the visualizer."""
        self.is_running = False


# Test code
if __name__ == "__main__":
    # Create test queues
    audio_queue = Queue()
    feedback_queue = Queue()
    
    # Add some test data
    import threading
    
    def generate_test_audio():
        """Generate test audio data."""
        time.sleep(1)
        while True:
            # Generate sine wave
            frequency = 440  # A note
            duration = 0.1
            sample_rate = 16000
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples)
            audio = np.sin(2 * np.pi * frequency * t) * 10000
            audio = audio.astype(np.int16)
            
            audio_queue.put(audio.tobytes())
            time.sleep(0.1)
    
    def generate_test_feedback():
        """Generate test feedback messages."""
        time.sleep(3)
        messages = [
            "키워드 검출: '그니까'",
            "발화 속도가 빠릅니다",
            "비속어 검출",
            "문법 오류 발견"
        ]
        
        for i, msg in enumerate(messages):
            time.sleep(3)
            feedback_queue.put({"message": msg, "type": "info"})
    
    # Start test threads
    audio_thread = threading.Thread(target=generate_test_audio, daemon=True)
    feedback_thread = threading.Thread(target=generate_test_feedback, daemon=True)
    
    audio_thread.start()
    feedback_thread.start()
    
    # Run visualizer
    visualizer = VoiceVisualizer(audio_queue, feedback_queue)
    visualizer.run()

