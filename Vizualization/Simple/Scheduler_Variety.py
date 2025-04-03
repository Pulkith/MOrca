import pygame
import random
import math
import time
from collections import deque

# Initialize pygame
pygame.init()

# Colors
BACKGROUND = (15, 25, 35)
CORE_BG = (30, 40, 50)
CORE_BORDER = (60, 70, 80)
TEXT_COLOR = (220, 220, 220)
TASK_COLORS = [
    (52, 152, 219),  # Blue
    (155, 89, 182),  # Purple
    (46, 204, 113),  # Green
    (231, 76, 60),   # Red
    (241, 196, 15),  # Yellow
    (230, 126, 34),  # Orange
    (26, 188, 156),  # Turquoise
    (211, 84, 0),    # Dark Orange
    (41, 128, 185),  # Dark Blue
    (142, 68, 173),  # Dark Purple
]
COMPLETED_COLOR = (100, 100, 100, 100)
BLOCKED_COLOR = (200, 60, 60)
HIGHLIGHT_COLOR = (255, 255, 255, 100)

# Screen dimensions
WIDTH, HEIGHT = 1200, 800
CORES = 8

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced CPU Core Scheduler Visualization")

# Fonts
font_small = pygame.font.SysFont("Arial", 14)
font_medium = pygame.font.SysFont("Arial", 18)
font_large = pygame.font.SysFont("Arial", 24)
font_title = pygame.font.SysFont("Arial", 32, bold=True)

class Task:
    def __init__(self, task_id, priority):
        self.id = task_id
        self.priority = priority  # 1-10, 10 being highest
        self.color = random.choice(TASK_COLORS)
        self.total_time = random.randint(2000, 10000)  # ms
        self.time_left = self.total_time
        self.blocked = False
        self.blocked_time = 0
        self.created_time = pygame.time.get_ticks()
        self.waiting_time = 0
        self.running_time = 0
        # Visual properties for animations
        self.pulse = 0
        self.pulse_direction = 1
        self.highlight = 0
        
    def update(self, delta_time, is_running=False):
        # Update pulse animation
        self.pulse += 0.05 * self.pulse_direction
        if self.pulse > 1.0 or self.pulse < 0.0:
            self.pulse_direction *= -1
            
        # Update highlight fade
        if self.highlight > 0:
            self.highlight -= 0.05
            
        # Update waiting time
        if not is_running and not self.blocked:
            self.waiting_time += delta_time
            
        # Update running time
        if is_running and not self.blocked:
            self.running_time += delta_time
            
        # Random chance to become blocked/unblocked
        if is_running and random.random() < 0.0005:
            self.blocked = not self.blocked
            if self.blocked:
                self.blocked_time = random.randint(1000, 3000)
                self.highlight = 1.0  # Highlight when blocked
            else:
                self.highlight = 1.0  # Highlight when unblocked
                
        # Update blocked time
        if self.blocked:
            self.blocked_time -= delta_time
            if self.blocked_time <= 0:
                self.blocked = False
                self.highlight = 1.0  # Highlight when unblocked
                
    def get_time_slice(self):
        # Higher priority tasks get more time before interruption
        return self.priority * 100  # 100-1000ms based on priority
        
    def get_progress(self):
        return 1.0 - (self.time_left / self.total_time)
        
    def is_complete(self):
        return self.time_left <= 0

class Core:
    def __init__(self, core_id, x, y, width, height):
        self.id = core_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.current_task = None
        self.task_start_time = 0
        self.utilization = 0
        self.utilization_history = [0] * 60  # Last 60 frames
        self.particles = []
        
    def assign_task(self, task):
        self.current_task = task
        self.task_start_time = pygame.time.get_ticks()
        # Create particles for visual effect
        for _ in range(10):
            self.particles.append({
                'x': self.x + self.width//2,
                'y': self.y + self.height//2,
                'dx': random.uniform(-2, 2),
                'dy': random.uniform(-2, 2),
                'life': 30,
                'color': task.color
            })
        
    def update(self, delta_time, task_queue):
        # Update current task
        if self.current_task:
            self.current_task.update(delta_time, True)
            
            # If task is blocked, update utilization accordingly
            if self.current_task.blocked:
                self.utilization = 0.2  # Low utilization when blocked
            else:
                self.utilization = 0.8 + random.uniform(-0.1, 0.2)  # High utilization when running
                
            # Check if time slice is completed
            time_elapsed = pygame.time.get_ticks() - self.task_start_time
            time_slice = self.current_task.get_time_slice()
            
            if not self.current_task.blocked:
                self.current_task.time_left -= delta_time
            
            # Task completed or time slice expired
            if (self.current_task.is_complete() or time_elapsed > time_slice) and not self.current_task.blocked:
                if self.current_task.is_complete():
                    scheduler.completed_tasks.append(self.current_task)
                    # Create completion particles
                    for _ in range(20):
                        self.particles.append({
                            'x': self.x + self.width//2,
                            'y': self.y + self.height//2,
                            'dx': random.uniform(-3, 3),
                            'dy': random.uniform(-3, 3),
                            'life': 40,
                            'color': (255, 255, 255)
                        })
                else:
                    # Put task back in queue with a slight priority boost for fairness
                    self.current_task.priority = min(10, self.current_task.priority + 0.5)
                    task_queue.append(self.current_task)
                    
                self.current_task = None
                
        else:
            self.utilization = 0.1 + random.uniform(0, 0.1)  # Idle with some background activity
            
        # Update utilization history
        self.utilization_history.pop(0)
        self.utilization_history.append(self.utilization)
        
        # Update particles
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                i += 1

    def draw(self, surface):
        # Draw core background with pulsing effect
        pygame.draw.rect(surface, CORE_BG, (self.x, self.y, self.width, self.height), border_radius=10)
        pygame.draw.rect(surface, CORE_BORDER, (self.x, self.y, self.width, self.height), 2, border_radius=10)
        
        # Draw utilization graph
        graph_height = 40
        graph_y = self.y + self.height - graph_height - 10
        for i, util in enumerate(self.utilization_history):
            bar_height = int(util * graph_height)
            bar_color = (
                int(52 + 179 * util),  # More red with higher utilization
                int(152 - 92 * util),  # Less green with higher utilization
                int(219 - 119 * util)   # Less blue with higher utilization
            )
            bar_width = (self.width - 20) / len(self.utilization_history)
            pygame.draw.rect(surface, bar_color, 
                            (self.x + 10 + i * bar_width, 
                             graph_y + graph_height - bar_height, 
                             bar_width, bar_height))
        
        # Draw core ID
        core_text = font_medium.render(f"Core {self.id}", True, TEXT_COLOR)
        surface.blit(core_text, (self.x + 10, self.y + 10))
        
        # Draw current task if any
        if self.current_task:
            task_height = 60
            task_y = self.y + 40
            
            # Task background with pulse effect
            pulse_factor = 0.1 * math.sin(self.current_task.pulse * math.pi)
            task_color = list(self.current_task.color)
            
            # Add highlight effect
            if self.current_task.highlight > 0:
                for i in range(3):
                    task_color[i] = min(255, task_color[i] + int(100 * self.current_task.highlight))
            
            # Draw task with progress bar
            progress = self.current_task.get_progress()
            pygame.draw.rect(surface, task_color, 
                            (self.x + 10, task_y, self.width - 20, task_height), 
                            border_radius=5)
            
            # Progress indicator
            pygame.draw.rect(surface, (255, 255, 255, 100), 
                            (self.x + 10, task_y, int((self.width - 20) * progress), task_height), 
                            border_radius=5)
            
            # Blocked indicator
            if self.current_task.blocked:
                block_surf = pygame.Surface((self.width - 20, task_height), pygame.SRCALPHA)
                block_surf.fill((200, 50, 50, 150))
                surface.blit(block_surf, (self.x + 10, task_y))
                blocked_text = font_medium.render("BLOCKED", True, (255, 255, 255))
                surface.blit(blocked_text, (
                    self.x + (self.width - blocked_text.get_width()) // 2, 
                    task_y + (task_height - blocked_text.get_height()) // 2
                ))
            
            # Task ID and info
            task_id_text = font_medium.render(f"Task {self.current_task.id}", True, (255, 255, 255))
            surface.blit(task_id_text, (self.x + 15, task_y + 5))
            
            # Priority indicator
            priority_text = font_small.render(f"Priority: {self.current_task.priority:.1f}", True, (255, 255, 255))
            surface.blit(priority_text, (self.x + 15, task_y + 25))
            
            # Progress percentage
            progress_text = font_small.render(f"{int(progress * 100)}%", True, (255, 255, 255))
            surface.blit(progress_text, (self.x + self.width - 50, task_y + 25))
            
            # Time slice indicator
            time_elapsed = pygame.time.get_ticks() - self.task_start_time
            time_slice = self.current_task.get_time_slice()
            slice_progress = min(1.0, time_elapsed / time_slice)
            
            pygame.draw.rect(surface, (80, 80, 80), 
                            (self.x + 10, task_y + task_height + 5, self.width - 20, 5), 
                            border_radius=2)
            pygame.draw.rect(surface, (200, 200, 100), 
                            (self.x + 10, task_y + task_height + 5, 
                             int((self.width - 20) * slice_progress), 5), 
                            border_radius=2)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            color = list(p['color'])
            if len(color) == 3:
                color.append(alpha)
            else:
                color[3] = alpha
            pygame.draw.circle(surface, color, (int(p['x']), int(p['y'])), int(p['life'] / 10) + 1)

class Scheduler:
    def __init__(self):
        self.cores = []
        self.task_queue = deque()
        self.completed_tasks = []
        self.next_task_id = 1
        self.task_generation_timer = 0
        self.task_counter = {
            "created": 0,
            "completed": 0,
            "waiting": 0
        }
        
        # Initialize cores
        core_width = 250
        core_height = 180
        margin = 20
        cores_per_row = 4
        for i in range(CORES):
            row = i // cores_per_row
            col = i % cores_per_row
            x = 50 + col * (core_width + margin)
            y = 100 + row * (core_height + margin)
            self.cores.append(Core(i, x, y, core_width, core_height))
            
        # Create initial tasks
        for _ in range(20):
            self.generate_task()
    
    def generate_task(self):
        priority = random.randint(1, 10)
        task = Task(self.next_task_id, priority)
        self.task_queue.append(task)
        self.next_task_id += 1
        self.task_counter["created"] += 1
    
    def update(self, delta_time):
        # Generate new tasks periodically
        self.task_generation_timer += delta_time
        if self.task_generation_timer > 500:  # Every 500ms
            self.task_generation_timer = 0
            if random.random() < 0.7 and len(self.task_queue) < 30:  # 70% chance to generate a task
                self.generate_task()
        
        # Sort queue by priority (with some randomness)
        task_list = list(self.task_queue)
        for task in task_list:
            task.update(delta_time, False)
        
        # Randomize a bit for visual interest, but respect priorities generally
        def priority_with_noise(task):
            return task.priority + random.uniform(-1, 1)
            
        task_list.sort(key=priority_with_noise, reverse=True)
        self.task_queue = deque(task_list)
        
        # Assign tasks to idle cores
        for core in self.cores:
            if core.current_task is None and self.task_queue:
                # Find first non-blocked task
                task_found = False
                for _ in range(len(self.task_queue)):
                    potential_task = self.task_queue.popleft()
                    if not potential_task.blocked:
                        core.assign_task(potential_task)
                        task_found = True
                        break
                    else:
                        self.task_queue.append(potential_task)
                        
                if not task_found and self.task_queue:
                    # If all tasks are blocked, just assign the first one anyway
                    core.assign_task(self.task_queue.popleft())
            
            # Update core
            core.update(delta_time, self.task_queue)
        
        # Update task counters
        self.task_counter["completed"] = len(self.completed_tasks)
        self.task_counter["waiting"] = len(self.task_queue)
                
    def draw(self, surface):
        # Draw title
        title = font_title.render("CPU Core Scheduler Visualization", True, TEXT_COLOR)
        surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
        
        # Draw cores
        for core in self.cores:
            core.draw(surface)
            
        # Draw task queue
        queue_x = 50
        queue_y = 500
        queue_width = WIDTH - 100
        queue_height = 100
        
        # Queue background
        pygame.draw.rect(surface, (25, 35, 45), (queue_x, queue_y, queue_width, queue_height), border_radius=10)
        pygame.draw.rect(surface, (50, 60, 70), (queue_x, queue_y, queue_width, queue_height), 2, border_radius=10)
        
        # Queue title
        queue_title = font_large.render("Task Queue", True, TEXT_COLOR)
        surface.blit(queue_title, (queue_x + 10, queue_y - 30))
        
        # Draw tasks in queue
        task_size = 30
        max_visible = min(20, len(self.task_queue))
        for i in range(max_visible):
            task = list(self.task_queue)[i]
            task_x = queue_x + 10 + i * (task_size + 5)
            task_y = queue_y + (queue_height - task_size) // 2
            
            # Task color with pulse effect
            pulse_factor = 0.2 * math.sin(task.pulse * math.pi)
            task_color = list(task.color)
            
            # Draw task
            pygame.draw.rect(surface, task_color, (task_x, task_y, task_size, task_size), border_radius=5)
            
            # Priority indicator (height of inner bar)
            priority_height = int(task.priority / 10 * task_size)
            pygame.draw.rect(surface, (255, 255, 255, 100), 
                            (task_x, task_y + task_size - priority_height, 
                             task_size, priority_height), 
                            border_radius=5)
            
            # Blocked indicator
            if task.blocked:
                block_surf = pygame.Surface((task_size, task_size), pygame.SRCALPHA)
                block_surf.fill((200, 50, 50, 150))
                surface.blit(block_surf, (task_x, task_y))
                pygame.draw.line(surface, (255, 255, 255), 
                                (task_x, task_y), 
                                (task_x + task_size, task_y + task_size), 2)
                pygame.draw.line(surface, (255, 255, 255), 
                                (task_x + task_size, task_y), 
                                (task_x, task_y + task_size), 2)
                
        # If queue is too long, indicate more tasks with dots
        if len(self.task_queue) > max_visible:
            for i in range(3):
                dot_x = queue_x + 10 + max_visible * (task_size + 5) + i * 10
                dot_y = queue_y + queue_height // 2
                pygame.draw.circle(surface, TEXT_COLOR, (dot_x, dot_y), 3)
                
        # Queue count
        queue_count = font_medium.render(f"{len(self.task_queue)} tasks waiting", True, TEXT_COLOR)
        surface.blit(queue_count, (queue_x + queue_width - queue_count.get_width() - 10, queue_y + 10))
        
        # Draw completed tasks
        completed_x = 50
        completed_y = 620
        completed_width = WIDTH - 100
        completed_height = 60
        
        # Completed background
        pygame.draw.rect(surface, (25, 35, 45), 
                        (completed_x, completed_y, completed_width, completed_height), 
                        border_radius=10)
        pygame.draw.rect(surface, (50, 60, 70), 
                        (completed_x, completed_y, completed_width, completed_height), 
                        2, border_radius=10)
        
        # Completed title
        completed_title = font_large.render("Completed Tasks", True, TEXT_COLOR)
        surface.blit(completed_title, (completed_x + 10, completed_y - 30))
        
        # Draw completed task indicators
        max_visible_completed = min(40, len(self.completed_tasks))
        for i in range(max_visible_completed):
            task = self.completed_tasks[-(i+1)]  # Start from most recent
            task_x = completed_x + 10 + i * 15
            task_y = completed_y + (completed_height - 40) // 2
            
            # Task color with reduced opacity
            task_color = list(task.color[:3]) + [150]  # Add alpha
            
            # Draw completed task marker
            pygame.draw.rect(surface, task_color, (task_x, task_y, 10, 40), border_radius=3)
        
        # Completed count
        completed_count = font_medium.render(f"{len(self.completed_tasks)} tasks completed", 
                                           True, TEXT_COLOR)
        surface.blit(completed_count, 
                   (completed_x + completed_width - completed_count.get_width() - 10, 
                    completed_y + 10))
        
        # Draw stats
        stats_x = 50
        stats_y = 700
        stats_width = WIDTH - 100
        stats_height = 60
        
        # Stats background
        pygame.draw.rect(surface, (25, 35, 45), 
                        (stats_x, stats_y, stats_width, stats_height), 
                        border_radius=10)
        pygame.draw.rect(surface, (50, 60, 70), 
                        (stats_x, stats_y, stats_width, stats_height), 
                        2, border_radius=10)
        
        # Calculate some stats
        active_cores = sum(1 for core in self.cores if core.current_task is not None)
        blocked_tasks = sum(1 for core in self.cores 
                          if core.current_task is not None and core.current_task.blocked)
        blocked_in_queue = sum(1 for task in self.task_queue if task.blocked)
        
        # Display stats
        stats = [
            f"Active Cores: {active_cores}/{CORES}",
            f"Blocked Tasks: {blocked_tasks + blocked_in_queue}",
            f"Tasks Created: {self.task_counter['created']}",
            f"Tasks Completed: {self.task_counter['completed']}",
            f"Tasks Waiting: {self.task_counter['waiting']}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = font_medium.render(stat, True, TEXT_COLOR)
            surface.blit(stat_text, (stats_x + 20 + i * (stats_width / len(stats)), stats_y + 20))

# Create scheduler
scheduler = Scheduler()

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear screen
    screen.fill(BACKGROUND)
    
    # Calculate delta time
    delta_time = clock.tick(60)
    
    # Update scheduler
    scheduler.update(delta_time)
    
    # Draw scheduler
    scheduler.draw(screen)
    
    # Update display
    pygame.display.flip()

# Quit pygame
pygame.quit()