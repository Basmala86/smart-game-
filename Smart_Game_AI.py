import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import heapq
import math
from collections import deque

# Game configuration
GRID_SIZE = 20
CELL_SIZE = 25
AGENT_COLOR = "blue"
TARGET_COLOR = "green"
OBSTACLE_COLOR = "red"
TERRAIN_COLORS = {
    "water": "#0066cc",
    "mountain": "#666666",
    "forest": "#009933"
}

class SmartGame:
    def __init__(self, master):
        self.master = master
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.agent_pos = (0, 0)
        self.targets = [(GRID_SIZE-1, GRID_SIZE-1)]
        self.obstacles = set()
        self.terrain = {}
        self.running = False
        self.score = 0
        self.steps = 0
        self.current_algorithm = None
        self.success_rate = []
        self.path_history = []  # For visualization

        # AI Models
        self.perceptron = make_pipeline(StandardScaler(), Perceptron(max_iter=1000))
        self.svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
        self.initialize_models()

        # ACO Parameters
        self.pheromones: npt.NDArray[np.float64] = np.ones((GRID_SIZE, GRID_SIZE)) * 0.1
        self.evaporation_rate = 0.05
        self.alpha = 1.2
        self.beta = 2.5

        # MCTS Parameters
        self.exploration_weight = 1.0
        self.simulation_depth = 15

        # RL Parameters
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # State-action values
        self.epsilon = 0.2
        self.alpha_rl = 0.1
        self.gamma = 0.9

        # Neural Evolution Parameters
        self.population_size = 10
        self.mutation_rate = 0.1
        self.generation = 0

        # GUI Setup
        self.create_gui()
        self.draw_grid()

    def create_gui(self):
        # Main canvas
        self.canvas = tk.Canvas(self.master, width=GRID_SIZE*CELL_SIZE, 
                              height=GRID_SIZE*CELL_SIZE)
        self.canvas.pack(side=tk.LEFT)

        # Control panel
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.RIGHT, padx=10)

        # Game info
        info_frame = tk.Frame(control_frame)
        info_frame.pack(fill=tk.X)

        self.score_label = tk.Label(info_frame, text=f"Score: {self.score}")
        self.score_label.pack(side=tk.LEFT, padx=5)

        self.steps_label = tk.Label(info_frame, text=f"Steps: {self.steps}")
        self.steps_label.pack(side=tk.LEFT, padx=5)

        # Algorithm selection
        algo_frame = tk.LabelFrame(control_frame, text="Algorithm")
        algo_frame.pack(pady=5, fill=tk.X)

        self.algo_var = tk.StringVar()
        algo_menu = ttk.Combobox(algo_frame, textvariable=self.algo_var,
                                values=["PSO", "ACO", "Perceptron", "SVM", 
                                        "A*", "MCTS", "Q-Learning", "Neural Evolution"])
        algo_menu.pack(pady=5, fill=tk.X)
        tk.Label(algo_frame, text="Revolutionary algorithms: A*, MCTS, Q-Learning, Neural Evolution", 
                font=("Arial", 8, "italic")).pack()

        # Terrain tools
        self.terrain_var = tk.StringVar()
        terrain_frame = tk.LabelFrame(control_frame, text="Terrain Tools")
        terrain_frame.pack(pady=5, fill=tk.X)

        for terrain in ["water", "mountain", "forest"]:
            tk.Radiobutton(terrain_frame, text=terrain, variable=self.terrain_var,
                          value=terrain).pack(anchor=tk.W)

        # Control buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=5, fill=tk.X)

        controls = [
            ("Start", self.start_game),
            ("Reset", self.reset_game)
        ]
        for text, cmd in controls:
            tk.Button(button_frame, text=text, command=cmd, 
                     width=6).pack(side=tk.LEFT, padx=2)

        map_frame = tk.Frame(control_frame)
        map_frame.pack(pady=5, fill=tk.X)

        map_controls = [
            ("Add Target", self.add_target),
            ("Random Maze", self.generate_maze),
            ("Clear Terrain", self.clear_terrain)
        ]
        for text, cmd in map_controls:
            tk.Button(map_frame, text=text, command=cmd).pack(fill=tk.X, pady=2)

        # Speed control
        self.speed_var = tk.DoubleVar(value=50)
        tk.Scale(control_frame, label="Speed", variable=self.speed_var,
                from_=1, to=100, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # Visualization options
        visual_frame = tk.LabelFrame(control_frame, text="Visualization")
        visual_frame.pack(pady=5, fill=tk.X)

        self.show_path_var = tk.BooleanVar(value=True)
        tk.Checkbutton(visual_frame, text="Show Path History", 
                      variable=self.show_path_var).pack(anchor=tk.W)

        # Event bindings
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.handle_drag)

    def initialize_models(self):
        X = []
        y = []
        for dy in range(-GRID_SIZE+1, GRID_SIZE):
            for dx in range(-GRID_SIZE+1, GRID_SIZE):
                if dy == 0 and dx == 0:
                    continue

                # Enhanced training with path awareness
                if self.is_axis_clear(dy, dx):
                    if abs(dy) > abs(dx):
                        direction = 0 if dy > 0 else 1
                    else:
                        direction = 2 if dx > 0 else 3
                else:
                    direction = random.randint(0, 3)

                # Extended feature set
                X.append([dy, dx, abs(dy), abs(dx)])
                y.append(direction)

        self.perceptron.fit(X, y)
        self.svm.fit(X, y)

    def is_axis_clear(self, dy, dx):
        """Check if general direction has fewer obstacles"""
        steps = max(abs(dy), abs(dx))
        if steps == 0:
            return True

        step_y = dy/steps
        step_x = dx/steps

        # Sample points along the direction
        for i in range(1, int(steps)+1):
            y = int(self.agent_pos[0] + i*step_y)
            x = int(self.agent_pos[1] + i*step_x)
            if (y, x) in self.obstacles:
                return False
        return True

    def draw_grid(self):
        self.canvas.delete("all")

        # Draw grid cells with appropriate colors
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = "white"
                if (i, j) in self.path_history and self.show_path_var.get():
                    color = "#ffcc99"  # Light orange for path history

                if (i, j) in self.terrain:
                    color = TERRAIN_COLORS[self.terrain[(i, j)][0]]

                if (i, j) in self.obstacles:
                    color = OBSTACLE_COLOR

                if (i, j) in self.targets:
                    color = TARGET_COLOR

                if (i, j) == self.agent_pos:
                    color = AGENT_COLOR

                # ACO pheromone visualization
                if color == "white" and self.current_algorithm == "ACO":
                    intensity = min(255, int(255 * self.pheromones[i, j]))
                    color = f"#{intensity:02x}{intensity:02x}ff"

                self.canvas.create_rectangle(
                    j*CELL_SIZE, i*CELL_SIZE,
                    (j+1)*CELL_SIZE, (i+1)*CELL_SIZE,
                    fill=color, outline="black"
                )

                # Draw Q-values if using Q-learning
                if self.current_algorithm == "Q-Learning" and (i, j) not in self.obstacles:
                    self.draw_q_values(i, j)

        # Update info labels
        self.steps_label.config(text=f"Steps: {self.steps}")

    def draw_q_values(self, i, j):
        """Draw directional indicators for Q-values"""
        q_vals = self.q_table[i, j]
        max_q = np.max(q_vals)

        if max_q > 0:
            # Draw arrows for directions with significant Q-values
            directions = [(0, -0.5, 0, 0.5), (-0.5, 0, 0.5, 0), (0, 0.5, 0, -0.5), (0.5, 0, -0.5, 0)]
            center_x = j * CELL_SIZE + CELL_SIZE/2
            center_y = i * CELL_SIZE + CELL_SIZE/2

            for d, (dx1, dy1, dx2, dy2) in enumerate(directions):
                q_val = q_vals[d]
                if q_val > 0:
                    # Calculate arrow size based on Q-value
                    strength = min(1.0, q_val / max_q)
                    arrow_len = strength * CELL_SIZE * 0.4

                    self.canvas.create_line(
                        center_x, center_y,
                        center_x + dx1 * arrow_len, center_y + dy1 * arrow_len,
                        arrow=tk.LAST, width=max(1, int(strength * 3)),
                        fill="#333333"
                    )

    def handle_click(self, event):
        if self.running:
            return

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        pos = (row, col)

        if self.terrain_var.get():
            self.terrain[pos] = (self.terrain_var.get(), random.uniform(1.2, 3))
        elif pos not in self.targets + [self.agent_pos]:
            if pos in self.obstacles:
                self.obstacles.remove(pos)
            else:
                self.obstacles.add(pos)
        self.draw_grid()

    def handle_drag(self, event):
        self.handle_click(event)

    def calculate_move_cost(self, pos):
        """Calculate movement cost based on terrain"""
        return self.terrain.get(pos, (None, 1))[1]

    def update_score(self, delta):
        self.score += delta
        self.score_label.config(text=f"Score: {self.score}")

    def get_valid_moves(self, pos):
        """Get all valid adjacent positions"""
        moves = [(pos[0]+1, pos[1]), (pos[0]-1, pos[1]),
                (pos[0], pos[1]+1), (pos[0], pos[1]-1)]
        return [m for m in moves if 
                0 <= m[0] < GRID_SIZE and
                0 <= m[1] < GRID_SIZE and
                m not in self.obstacles]

    def run_pso(self):
        """Run Particle Swarm Optimization algorithm"""
        particles = [self.agent_pos] * 10
        best_pos = self.agent_pos

        for _ in range(100):
            if not self.running:
                break
            new_particles = []
            for p in particles:
                dx = np.sign(self.targets[0][1] - p[1])
                dy = np.sign(self.targets[0][0] - p[0])
                new_p = (p[0] + dy, p[1] + dx)
                if self.get_valid_moves(p) and self.is_valid_move(new_p):
                    new_particles.append(new_p)
                    if self.distance(new_p) < self.distance(best_pos):
                        best_pos = new_p
            particles = new_particles
            self.agent_pos = best_pos
            self.path_history.append(self.agent_pos)
            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(100 - self.speed_var.get()))
            if self.agent_pos in self.targets:
                self.update_score(100)
                break
        self.running = False

    def run_aco(self):
        """Run Ant Colony Optimization algorithm"""
        self.reset_pheromones()
        path = []
        current_pos = self.agent_pos
        visited = set()

        for _ in range(200):
            if not self.running:
                break
            if current_pos in self.targets:
                self.deposit_pheromones(path)
                self.update_score(100)
                break

            moves = self.get_valid_moves(current_pos)
            moves = [m for m in moves if m not in visited]

            if not moves:
                if path:
                    current_pos = path.pop()
                    visited.remove(current_pos)
                    continue
                else:
                    break

            probabilities = []
            total = 0
            for move in moves:
                heuristic = 1 / (self.distance(move) if self.distance(move) != 0 else 1)
                row = int(move[0])
                col = int(move[1])
                pheromone = self.pheromones[row, col]
                cost = self.calculate_move_cost(move)
                score = (pheromone ** self.alpha) * (heuristic ** self.beta) / cost
                probabilities.append(score)
                total += score

            if total == 0:
                chosen = random.choice(moves)
            else:
                probabilities = [p/total for p in probabilities]
                chosen = np.random.choice(len(moves), p=probabilities)

            next_pos = moves[chosen]
            path.append(current_pos)
            visited.add(current_pos)
            current_pos = next_pos
            self.agent_pos = current_pos
            self.path_history.append(self.agent_pos)

            row = int(current_pos[0])
            col = int(current_pos[1])
            self.pheromones[row, col] += 0.1
            self.evaporate_pheromones()

            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(100 - self.speed_var.get()))

        self.running = False

    def run_perceptron(self):
        """Run Perceptron-based navigation"""
        max_steps = 200
        visited = set()

        for _ in range(max_steps):
            if not self.running:
                break

            moves = self.get_valid_moves(self.agent_pos)
            if not moves:
                break

            target = min(self.targets, key=lambda t: self.distance(t, self.agent_pos))
            dy = target[0] - self.agent_pos[0]
            dx = target[1] - self.agent_pos[1]

            direction = self.perceptron.predict([[dy, dx, abs(dy), abs(dx)]])[0]
            new_pos = self.calculate_new_pos(direction)

            if new_pos not in moves or new_pos in visited:
                new_pos = random.choice(moves)

            self.agent_pos = new_pos
            self.path_history.append(self.agent_pos)
            visited.add(new_pos)

            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(100 - self.speed_var.get()))

            if self.agent_pos in self.targets:
                self.update_score(100)
                break

        self.running = False

    def run_svm(self):
        """Run SVM-based navigation"""
        max_steps = 200
        visited = set()
        success = False

        for _ in range(max_steps):
            if not self.running:
                break

            # Get all valid moves
            moves = self.get_valid_moves(self.agent_pos)
            if not moves:
                break

            # Find nearest target
            target = min(self.targets, key=lambda t: self.distance(t, self.agent_pos))
            dy = target[0] - self.agent_pos[0]
            dx = target[1] - self.agent_pos[1]

            # Get SVM's suggested direction
            direction = self.svm.predict([[dy, dx, abs(dy), abs(dx)]])[0]
            new_pos = self.calculate_new_pos(direction)

            # Enhanced fallback mechanism
            if new_pos not in moves or new_pos in visited:
                moves.sort(key=lambda m: (
                    self.distance(m) * self.calculate_move_cost(m),
                    random.random()
                ))
                new_pos = moves[0] if moves else self.agent_pos

            self.agent_pos = new_pos
            self.path_history.append(self.agent_pos)
            visited.add(new_pos)

            # Visual updates
            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(100 - self.speed_var.get()))

            if self.agent_pos in self.targets:
                self.update_score(100)
                success = True
                break

        # Track performance
        self.success_rate.append(1 if success else 0)
        if len(self.success_rate) > 10:
            print(f"SVM Success Rate (last 10): {sum(self.success_rate[-10:])/10:.0%}")
        self.running = False

    # === REVOLUTIONARY ALGORITHMS ===

    def run_a_star(self):
        """Run A* Pathfinding Algorithm"""
        start = self.agent_pos
        target = min(self.targets, key=lambda t: self.distance(t, start))

        # Priority queue for A*
        open_set = [(0, 0, start, [])]  # (f_score, g_score, pos, path)
        heapq.heapify(open_set)

        closed_set = set()
        step_counter = 0
        max_steps = 300

        while open_set and step_counter < max_steps and self.running:
            # Get node with lowest f-score
            f, g, current, path = heapq.heappop(open_set)

            if current in closed_set:
                continue

            # Path found
            if current in self.targets:
                # Animate solution path
                for pos in path + [current]:
                    if not self.running:
                        break
                    self.agent_pos = pos
                    self.path_history.append(pos)
                    self.steps += 1
                    self.draw_grid()
                    self.master.update()
                    self.master.after(int(50 - self.speed_var.get()/2))

                self.update_score(100)
                self.running = False
                return

            closed_set.add(current)

            # Check all neighbors
            for neighbor in self.get_valid_moves(current):
                if neighbor in closed_set:
                    continue

                # Calculate scores
                movement_cost = self.calculate_move_cost(neighbor)
                g_score = g + movement_cost
                h_score = self.distance(neighbor, target)
                f_score = g_score + h_score

                # Add to open set
                heapq.heappush(open_set, (f_score, g_score, neighbor, path + [current]))

            # Visualization for exploration
            if step_counter % 2 == 0:  # Reduce visualization frequency for speed
                self.agent_pos = current
                self.path_history.append(current)
                self.steps += 1
                self.draw_grid()
                self.master.update()
                self.master.after(int(100 - self.speed_var.get()))

            step_counter += 1

        # No path found
        messagebox.showinfo("A* Result", "No path found or maximum steps reached")
        self.running = False

    def run_mcts(self):
        """Run Monte Carlo Tree Search"""
        class MCTSNode:
            def __init__(self, state, parent=None, action=None):
                self.state = state
                self.parent = parent
                self.action = action
                self.children = []
                self.visits = 0
                self.value = 0

        max_iterations = 200
        root = MCTSNode(self.agent_pos)

        for iteration in range(max_iterations):
            if not self.running:
                break

            # Selection
            node = root
            path = [node.state]

            while node.children and not self.is_terminal(node.state):
                # Select child with highest UCB value
                node = max(node.children, key=lambda n: self.ucb_score(n))
                path.append(node.state)

            # Expansion
            if not self.is_terminal(node.state):
                actions = self.get_valid_moves(node.state)
                for action in actions:
                    if not any(child.state == action for child in node.children):
                        node.children.append(MCTSNode(action, node, action))

                if node.children:
                    node = random.choice(node.children)
                    path.append(node.state)

            # Simulation
            state = node.state
            simulation_path = []
            for _ in range(self.simulation_depth):
                if self.is_terminal(state):
                    break

                moves = self.get_valid_moves(state)
                if not moves:
                    break

                # Prioritize moves closer to target
                moves.sort(key=lambda m: self.distance(m))
                # Add some exploration
                if random.random() < 0.3:
                    state = random.choice(moves)
                else:
                    state = moves[0]

                simulation_path.append(state)

            # Backpropagation
            reward = 1.0 / (1.0 + self.distance(state))  # Reward inversely proportional to distance
            if state in self.targets:
                reward = 1.0  # Maximum reward for reaching target

            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

            # Visualization (less frequent for efficiency)
            if iteration % 5 == 0:
                # Show the currently explored path
                for pos in path:
                    self.agent_pos = pos
                    self.path_history.append(pos)
                    self.steps += 1
                    self.draw_grid()
                    self.master.update()
                    self.master.after(int(100 - self.speed_var.get()))

                if self.agent_pos in self.targets:
                    self.update_score(100)
                    self.running = False
                    return

        # Take best path after iterations
        best_path = []
        node = root

        while node.children:
            node = max(node.children, key=lambda n: n.visits)
            best_path.append(node.state)

        # Animate the best found path
        for pos in best_path:
            if not self.running:
                break
            self.agent_pos = pos
            self.path_history.append(pos)
            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(50 - self.speed_var.get()/2))

            if self.agent_pos in self.targets:
                self.update_score(100)
                break

        self.running = False

    def ucb_score(self, node):
        """UCB1 score for MCTS node selection"""
        if node.visits == 0:
            return float('inf')

        exploitation = node.value / node.visits
        exploration = self.exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
        return exploitation + exploration

    def is_terminal(self, state):
        """Check if state is terminal (target reached)"""
        return state in self.targets

    def run_q_learning(self):
        """Run Q-Learning algorithm"""
        max_steps = 300
        success = False

        # Directions: Down(0), Up(1), Right(2), Left(3)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Initialize Q-table if necessary
        if np.sum(self.q_table) == 0:
            self.initialize_q_table()

        current_pos = self.agent_pos

        for step in range(max_steps):
            if not self.running:
                break

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Explore: random action
                valid_moves = self.get_valid_moves(current_pos)
                if not valid_moves:
                    break

                direction_options = []
                for i, (dy, dx) in enumerate(directions):
                    new_pos = (current_pos[0] + dy, current_pos[1] + dx)
                    if new_pos in valid_moves:
                        direction_options.append(i)

                if not direction_options:
                    break

                action = random.choice(direction_options)
            else:
                # Exploit: best action from Q-table
                action = np.argmax(self.q_table[current_pos[0], current_pos[1]])

                # Check if action is valid
                new_pos = (current_pos[0] + directions[action][0], 
                          current_pos[1] + directions[action][1])

                if not self.is_valid_move(new_pos):
                    # If invalid, choose from valid moves
                    valid_moves = self.get_valid_moves(current_pos)
                    if not valid_moves:
                        break

                    # Choose move with highest Q-value among valid moves
                    best_q = -float('inf')
                    best_action = 0

                    for i, (dy, dx) in enumerate(directions):
                        new_pos = (current_pos[0] + dy, current_pos[1] + dx)
                        if new_pos in valid_moves:
                            q_val = self.q_table[current_pos[0], current_pos[1], i]
                            if q_val > best_q:
                                best_q = q_val
                                best_action = i

                    action = best_action

            # Take action
            new_pos = (current_pos[0] + directions[action][0], 
                      current_pos[1] + directions[action][1])

            # Ensure the move is valid
            if not self.is_valid_move(new_pos):
                continue

            # Calculate reward
            reward = -0.1  # Small negative reward for each step
            if new_pos in self.targets:
                reward = 100  # Large reward for reaching target
            elif new_pos in self.terrain:
                terrain_type = self.terrain[new_pos][0]
                # Additional terrain-specific rewards
                if terrain_type == "water":
                    reward -= 0.5
                elif terrain_type == "mountain":
                    reward -= 1.0

            # Q-Learning update
            current_q = self.q_table[current_pos[0], current_pos[1], action]
            max_future_q = np.max(self.q_table[new_pos[0], new_pos[1]])

            new_q = (1 - self.alpha_rl) * current_q + self.alpha_rl * (reward + self.gamma * max_future_q)
            self.q_table[current_pos[0], current_pos[1], action] = new_q

            # Update position
            current_pos = new_pos
            self.agent_pos = current_pos
            self.path_history.append(current_pos)

            # Check for target
            if current_pos in self.targets:
                self.update_score(100)
                success = True
                break

            # Visualization
            self.steps += 1
            self.draw_grid()
            self.master.update()
            self.master.after(int(100 - self.speed_var.get()))

        if success:
            print("Q-Learning successfully reached target!")
            # Decrease exploration over time
            self.epsilon = max(0.05, self.epsilon * 0.95)

        self.running = False

    def initialize_q_table(self):
        """Initialize Q-table with simple heuristic values"""
        # Initialize Q-values based on distance to nearest target
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) in self.obstacles:
                    continue

                # Get directions that move towards target
                target = min(self.targets, key=lambda t: self.distance((i, j), t))
                dy = target[0] - i
                dx = target[1] - j

                # Set higher initial Q-values for directions moving toward target
                if dy > 0:  # Target is below current position
                    self.q_table[i, j, 0] = 0.5  # Down direction
                elif dy < 0:  # Target is above current position
                    self.q_table[i, j, 1] = 0.5  # Up direction

                if dx > 0:  # Target is right of current position
                    self.q_table[i, j, 2] = 0.5  # Right direction
                elif dx < 0:  # Target is left of current position
                    self.q_table[i, j, 3] = 0.5  # Left direction

    def run_neural_evolution(self):
        """Run Neural Evolution algorithm"""
        class SimpleNN:
            def __init__(self, weights=None):
                # Simple neural network with 1 hidden layer
                if weights is None:
                    # 6 input features, 8 hidden neurons, 4 outputs (directions)
                    self.weights1 = np.random.randn(6, 8) * 0.1
                    self.weights2 = np.random.randn(8, 4) * 0.1
                else:
                    self.weights1, self.weights2 = weights

            def forward(self, x):
                # Simple feedforward
                self.hidden = np.tanh(np.dot(x, self.weights1))
                self.output = np.tanh(np.dot(self.hidden, self.weights2))
                return self.output

            def get_weights(self):
                return (self.weights1, self.weights2)

            def mutate(self, rate):
                """Mutate weights with given rate"""
                mutation = np.random.randn(*self.weights1.shape) * rate
                self.weights1 += mutation

                mutation = np.random.randn(*self.weights2.shape) * rate
                self.weights2 += mutation

            def crossover(self, other):
                """Perform crossover with another network"""
                # Create masks for crossover (randomly select weights from either parent)
                mask1 = np.random.choice([0, 1], size=self.weights1.shape).astype(bool)
                mask2 = np.random.choice([0, 1], size=self.weights2.shape).astype(bool)

                # Create child weights
                child_w1 = np.copy(self.weights1)
                child_w1[mask1] = other.weights1[mask1]

                child_w2 = np.copy(self.weights2)
                child_w2[mask2] = other.weights2[mask2]

                return SimpleNN((child_w1, child_w2))

        # Create initial population
        population = [SimpleNN() for _ in range(self.population_size)]

        max_generations = 10
        max_steps_per_generation = 100

        for generation in range(max_generations):
            if not self.running:
                break

            fitnesses = []
            paths = []

            # Evaluate each network
            for network in population:
                if not self.running:
                    break

                # Reset for this evaluation
                current_pos = self.agent_pos
                steps = 0
                path = [current_pos]
                visited = set([current_pos])

                while steps < max_steps_per_generation:
                    # Extract features
                    target = min(self.targets, key=lambda t: self.distance(current_pos, t))
                    dy = target[0] - current_pos[0]
                    dx = target[1] - current_pos[1]

                    # Get surrounding terrain costs (simplified to binary obstacle/free)
                    surroundings = []
                    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                        if self.is_valid_move(pos):
                            surroundings.append(1.0)
                        else:
                            surroundings.append(0.0)

                    # Normalize inputs
                    dy_norm = dy / GRID_SIZE
                    dx_norm = dx / GRID_SIZE

                    # Create input vector
                    inputs = np.array([dy_norm, dx_norm, *surroundings])

                    # Get network output
                    outputs = network.forward(inputs)

                    # Choose action based on network output
                    valid_moves = self.get_valid_moves(current_pos)
                    if not valid_moves:
                        break

                    # Create a list of (direction_index, output_value) pairs
                    direction_values = [(i, outputs[i]) for i in range(4)]

                    # Sort by output value (highest first)
                    direction_values.sort(key=lambda x: x[1], reverse=True)

                    # Try directions in order of network preference
                    next_pos = None
                    for direction_idx, _ in direction_values:
                        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        dy, dx = directions[direction_idx]
                        pos = (current_pos[0] + dy, current_pos[1] + dx)

                        if pos in valid_moves and pos not in visited:
                            next_pos = pos
                            break

                    # If all preferred directions are invalid/visited, choose randomly from valid moves
                    if next_pos is None:
                        unvisited_moves = [m for m in valid_moves if m not in visited]
                        if unvisited_moves:
                            next_pos = random.choice(unvisited_moves)
                        elif valid_moves:  # Allow revisiting if necessary
                            next_pos = random.choice(valid_moves)
                        else:
                            break

                    # Move to next position
                    current_pos = next_pos
                    path.append(current_pos)
                    visited.add(current_pos)
                    steps += 1

                    # Check if target reached
                    if current_pos in self.targets:
                        break

                # Calculate fitness based on:
                # 1. Distance to target (closer = better)
                # 2. Path length (shorter = better)
                # 3. Reaching target (huge bonus)

                distance_to_target = self.distance(current_pos)
                path_penalty = len(path) * 0.1
                target_bonus = 1000 if current_pos in self.targets else 0

                fitness = 100 - distance_to_target - path_penalty + target_bonus
                fitnesses.append(fitness)
                paths.append(path)

            # Visualize best path from this generation
            best_idx = np.argmax(fitnesses)
            best_path = paths[best_idx]
            best_fitness = fitnesses[best_idx]

            # Visualize best path
            print(f"Generation {generation+1}: Best fitness = {best_fitness:.2f}")

            old_agent_pos = self.agent_pos
            for pos in best_path:
                if not self.running:
                    break
                self.agent_pos = pos
                self.path_history.append(pos)
                self.steps += 1
                self.draw_grid()
                self.master.update()
                self.master.after(int(100 - self.speed_var.get()))

                if self.agent_pos in self.targets:
                    self.update_score(100)
                    self.running = False
                    return

            # Reset agent position
            self.agent_pos = old_agent_pos

            # Create next generation
            next_population = []

            # Elitism: keep the best network
            best_network = population[best_idx]
            next_population.append(best_network)

            # Tournament selection + crossover
            while len(next_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(self.population_size), tournament_size)
                parent1_idx = max(tournament_indices, key=lambda i: fitnesses[i])

                tournament_indices = random.sample(range(self.population_size), tournament_size)
                parent2_idx = max(tournament_indices, key=lambda i: fitnesses[i])

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child = parent1.crossover(parent2)

                # Mutation
                child.mutate(self.mutation_rate)

                next_population.append(child)

            # Update population
            population = next_population

            # Update mutation rate (decrease over time)
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)

            # Check if we should stop
            if best_fitness > 900:  # Target reached with good path
                break

        self.running = False

    def distance(self, pos1, pos2=None):
        """Manhattan distance between positions"""
        if pos2 is None:
            # Find minimum distance to any target
            return min(abs(pos1[0] - t[0]) + abs(pos1[1] - t[1]) for t in self.targets)
        else:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_valid_move(self, pos):
        """Check if position is valid (in bounds and not obstacle)"""
        return (0 <= pos[0] < GRID_SIZE and 
                0 <= pos[1] < GRID_SIZE and 
                pos not in self.obstacles)

    def calculate_new_pos(self, direction):
        """Calculate new position based on direction index"""
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        return (self.agent_pos[0] + dirs[direction][0],
                self.agent_pos[1] + dirs[direction][1])

    def add_target(self):
        """Add random target to the grid"""
        if self.running:
            return

        empty = [
            (i,j) for i in range(GRID_SIZE) 
            for j in range(GRID_SIZE)
            if (i,j) not in self.targets 
            and (i,j) != self.agent_pos
            and (i,j) not in self.obstacles
        ]
        if empty:
            self.targets.append(random.choice(empty))
            self.draw_grid()

    def generate_maze(self):
        """Generate random maze with obstacles"""
        if self.running:
            return

        self.obstacles = set()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if random.random() < 0.25 and (i,j) not in self.targets and (i,j) != self.agent_pos:
                    self.obstacles.add((i,j))

        # Ensure maze is solvable using BFS
        if not self.is_solvable():
            messagebox.showinfo("Maze Generation", "Generated maze has no solution. Trying again.")
            self.generate_maze()
            return

        self.draw_grid()

    def is_solvable(self):
        """Check if the current maze has a path to the target"""
        start = self.agent_pos
        queue = deque([start])
        visited = set([start])

        while queue:
            current = queue.popleft()

            if current in self.targets:
                return True

            for neighbor in self.get_valid_moves(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def clear_terrain(self):
        """Clear all terrain from the grid"""
        if self.running:
            return

        self.terrain = {}
        self.draw_grid()

    def reset_pheromones(self):
        """Reset pheromone levels for ACO"""
        self.pheromones = np.ones((GRID_SIZE, GRID_SIZE)) * 0.1
        for obstacle in self.obstacles:
            row, col = int(obstacle[0]), int(obstacle[1])
            self.pheromones[row, col] = 0

    def evaporate_pheromones(self):
        """Evaporate pheromones for ACO"""
        self.pheromones *= (1 - self.evaporation_rate)

    def deposit_pheromones(self, path):
        """Deposit pheromones along a path for ACO"""
        for pos in path:
            row, col = int(pos[0]), int(pos[1])
            self.pheromones[row, col] += 1.0

    def start_game(self):
        """Start the game with selected algorithm"""
        if not self.running and self.algo_var.get():
            self.running = True
            self.path_history = []  # Clear path history
            self.current_algorithm = self.algo_var.get()

            # Convert algorithm name to method name and call it
            algorithm_map = {
                "PSO": "run_pso",
                "ACO": "run_aco",
                "Perceptron": "run_perceptron",
                "SVM": "run_svm",
                "A*": "run_a_star",
                "MCTS": "run_mcts",
                "Q-Learning": "run_q_learning",
                "Neural Evolution": "run_neural_evolution"
            }

            method_name = algorithm_map.get(self.current_algorithm)
            if method_name:
                getattr(self, method_name)()

            self.current_algorithm = None

    def reset_game(self):
        """Reset the game to initial state"""
        self.running = False
        self.agent_pos = (0, 0)
        self.targets = [(GRID_SIZE-1, GRID_SIZE-1)]
        self.obstacles = set()
        self.terrain = {}
        self.score = 0
        self.steps = 0
        self.path_history = []
        self.reset_pheromones()
        self.draw_grid()
        self.score_label.config(text="Score: 0")
        self.steps_label.config(text="Steps: 0")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Revolutionary AI Pathfinding Game")
    game = SmartGame(root)
    root.mainloop()