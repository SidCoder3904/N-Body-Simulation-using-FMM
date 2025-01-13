// fmm_visualization.cpp

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <deque>
#include <unistd.h>

// Include OpenGL headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;
const int num_particles = 1000;
const double EPSILON = 1e-5;
const int MAX_PARTICLES = 50; // Maximum particles per leaf node
const int MAX_HISTORY = 20; // Maximum particles per leaf node
const int MAX_DEPTH = 10;      // Maximum depth of the tree
const int P = 5;              // Order of the multipole expansion
double dt = 0.00005;
// Window dimensions
const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 1000;

// Structure to represent a particle
struct Particle {
    double x, y;    // Position
    double vx, vy;  // Velocity components
    double fx, fy;  // Force components
    double mass;    // Mass
    deque<pair<double, double>> history; // Store the last 50 positions

    Particle(double x_, double y_, double mass_) : x(x_), y(y_), vx(0.0), vy(0.0), fx(0.0), fy(0.0), mass(mass_) {}
};

// Class to represent a cell in the quadtree
class QuadTreeNode {
public:
    // Bounding box of the cell
    double x_min, x_max, y_min, y_max;
    // Center of the cell
    double x_center, y_center;
    // Level in the tree
    int depth;
    // Pointers to child nodes
    QuadTreeNode* children[4];
    // Particles contained in the cell (if leaf node)
    vector<Particle*> particles;
    // Multipole coefficients
    vector<complex<double>> multipole_coeffs;
    // Local expansion coefficients
    vector<complex<double>> local_coeffs;

    // Parent node
    QuadTreeNode* parent;

    // Interaction list for M2L translations
    vector<QuadTreeNode*> interaction_list;

    // Constructor
    QuadTreeNode(double x_min_, double x_max_, double y_min_, double y_max_, int depth_, QuadTreeNode* parent_)
        : x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_), depth(depth_), parent(parent_) {
        x_center = 0.5 * (x_min + x_max);
        y_center = 0.5 * (y_min + y_max);
        for (int i = 0; i < 4; ++i) {
            children[i] = nullptr;
        }
        multipole_coeffs.resize(P + 1, complex<double>(0.0, 0.0));
        local_coeffs.resize(P + 1, complex<double>(0.0, 0.0));
    }

    // Destructor to free child nodes
    ~QuadTreeNode() {
        for (int i = 0; i < 4; ++i) {
            delete children[i];
        }
    }

    // Insert a particle into the quadtree
    void insert_particle(Particle* p) {
        // If this node is not a leaf node, pass the particle to the appropriate child
        if (children[0] != nullptr) {
            int quadrant = get_quadrant(p);
            children[quadrant]->insert_particle(p);
            return;
        }

        // Add particle to this node
        particles.push_back(p);

        // If capacity exceeded and maximum depth not reached, subdivide
        if (particles.size() > MAX_PARTICLES && depth < MAX_DEPTH) {
            subdivide();
            // Reassign particles to children
            for (Particle* particle : particles) {
                int quadrant = get_quadrant(particle);
                children[quadrant]->insert_particle(particle);
            }
            particles.clear();
        }
    }

    // Subdivide the cell into four child quadrants
    void subdivide() {
        double x_mid = x_center;
        double y_mid = y_center;
        int next_depth = depth + 1;
        children[0] = new QuadTreeNode(x_min, x_mid, y_min, y_mid, next_depth, this); // SW
        children[1] = new QuadTreeNode(x_mid, x_max, y_min, y_mid, next_depth, this); // SE
        children[2] = new QuadTreeNode(x_min, x_mid, y_mid, y_max, next_depth, this); // NW
        children[3] = new QuadTreeNode(x_mid, x_max, y_mid, y_max, next_depth, this); // NE
    }

    // Determine which quadrant a particle belongs to
    int get_quadrant(Particle* p) {
        int quadrant = 0;
        if (p->x >= x_center) quadrant += 1; // East
        if (p->y >= y_center) quadrant += 2; // North
        return quadrant;
    }

    // Compute multipole expansions recursively
    void compute_multipole() {
        if (children[0] == nullptr) {
            // Leaf node: compute multipole expansion directly from particles
            for (Particle* p : particles) {
                complex<double> z(p->x - x_center, p->y - y_center);
                multipole_coeffs[0] += p->mass;
                for (int k = 1; k <= P; ++k) {
                    multipole_coeffs[k] += p->mass * pow(z, k);
                }
            }
        } else {
            // Internal node: aggregate multipole expansions from children
            for (int i = 0; i < 4; ++i) {
                children[i]->compute_multipole();
            }
            // Shift and combine children's multipole expansions
            for (int i = 0; i < 4; ++i) {
                QuadTreeNode* child = children[i];
                complex<double> z(child->x_center - x_center, child->y_center - y_center);
                for (int k = 0; k <= P; ++k) {
                    complex<double> shifted_coeff(0.0, 0.0);
                    for (int n = 0; n <= k; ++n) {
                        shifted_coeff += child->multipole_coeffs[n]
                                         * binomial_coefficient(k, n)
                                         * pow(z, k - n);
                    }
                    multipole_coeffs[k] += shifted_coeff;
                }
            }
        }
    }

    // Compute local expansions recursively
    void compute_local_expansion() {
        if (parent != nullptr) {
            // Translate parent's local expansion to this node
            complex<double> z(x_center - parent->x_center, y_center - parent->y_center);
            for (int k = 0; k <= P; ++k) {
                for (int n = k; n <= P; ++n) {
                    local_coeffs[k] += parent->local_coeffs[n]
                                       * binomial_coefficient(n, k)
                                       * pow(z, n - k);
                }
            }
        }

        // Build interaction list if not already built
        if (interaction_list.empty()) {
            interaction_list = get_interaction_list();
        }

        // Interact with nodes in the interaction list
        for (QuadTreeNode* node : interaction_list) {
            complex<double> dz(node->x_center - x_center, node->y_center - y_center);
            for (int n = 0; n <= P; ++n) {
                local_coeffs[n] += node->multipole_coeffs[n] * pow(dz, -n - 1);
            }
        }

        // Propagate local expansions to children
        if (children[0] != nullptr) {
            for (int i = 0; i < 4; ++i) {
                children[i]->compute_local_expansion();
            }
        }
    }

    // Evaluate the local expansions at the particles
    void evaluate() {
        if (children[0] == nullptr) {
            // Leaf node: evaluate local expansions and compute direct interactions
            for (Particle* p : particles) {
                complex<double> z(p->x - x_center, p->y - y_center);
                complex<double> force(0.0, 0.0);
                for (int k = 1; k <= P; ++k) {
                    force += local_coeffs[k] * (double)k * pow(z, k - 1);
                }
                p->fx += force.real();
                p->fy += force.imag();
                // Compute direct interactions with nearby particles
                vector<QuadTreeNode*> neighbors = get_neighbor_list();
                for (QuadTreeNode* node : neighbors) {
                    for (Particle* q : node->particles) {
                        if (p != q) {
                            double dx = q->x - p->x;
                            double dy = q->y - p->y;
                            double r2 = dx * dx + dy * dy + EPSILON;
                            double inv_r3 = q->mass / (r2 * sqrt(r2));
                            p->fx += inv_r3 * dx;
                            p->fy += inv_r3 * dy;
                        }
                    }
                }
            }
            for (size_t i = 0; i < particles.size(); ++i) {
                for (size_t j = 0; j < particles.size(); ++j) {
                    if (i != j) {
                        double dx = particles[j]->x - particles[i]->x;
                        double dy = particles[j]->y - particles[i]->y;
                        double r2 = dx * dx + dy * dy + EPSILON;
                        double inv_r3 = particles[j]->mass / (r2 * sqrt(r2));
                        particles[i]->fx += inv_r3 * dx;
                        particles[i]->fy += inv_r3 * dy;
                    }
                }
            }
    }
        else {
            // Recursively evaluate children
            for (int i = 0; i < 4; ++i) {
                children[i]->evaluate();
            }
        }
    }

    // Render the quadtree cells
    void render_cells() {
        // Draw the cell boundary
        glColor3f(0.5f, 0.5f, 0.5f); // Gray color
        glBegin(GL_LINE_LOOP);
        glVertex2f(x_min, y_min);
        glVertex2f(x_max, y_min);
        glVertex2f(x_max, y_max);
        glVertex2f(x_min, y_max);
        glEnd();

        // Recursively render child cells
        if (children[0] != nullptr) {
            for (int i = 0; i < 4; ++i) {
                children[i]->render_cells();
            }
        }
    }

private:
    // Compute binomial coefficient
    double binomial_coefficient(int n, int k) {
        if (k == 0 || k == n) return 1.0;
        if (k > n) return 0.0;
        double res = 1.0;
        for (int i = 1; i <= k; ++i) {
            res *= n--;
            res /= i;
        }
        return res;
    }

    // Get interaction list (well-separated nodes)
    vector<QuadTreeNode*> get_interaction_list() {
        vector<QuadTreeNode*> interaction_list;
        if (parent == nullptr) return interaction_list;

        vector<QuadTreeNode*> parent_neighbors = parent->get_neighbor_list();
        for (QuadTreeNode* neighbor : parent_neighbors) {
            if (neighbor->children[0] != nullptr) {
                for (int i = 0; i < 4; ++i) {
                    QuadTreeNode* child = neighbor->children[i];
                    if (child != nullptr && is_well_separated(child)) {
                        interaction_list.push_back(child);
                    }
                }
            } else {
                if (is_well_separated(neighbor)) {
                    interaction_list.push_back(neighbor);
                }
            }
        }

        // Include parent's interaction list
        vector<QuadTreeNode*> parent_interactions = parent->interaction_list;
        for (QuadTreeNode* node : parent_interactions) {
            if (node->children[0] != nullptr) {
                for (int i = 0; i < 4; ++i) {
                    QuadTreeNode* child = node->children[i];
                    if (child != nullptr && is_well_separated(child)) {
                        interaction_list.push_back(child);
                    }
                }
            } else {
                if (is_well_separated(node)) {
                    interaction_list.push_back(node);
                }
            }
        }

        // Remove duplicates
        sort(interaction_list.begin(), interaction_list.end());
        interaction_list.erase(unique(interaction_list.begin(), interaction_list.end()), interaction_list.end());

        return interaction_list;
    }

    // Get neighbor list (adjacent nodes)
    vector<QuadTreeNode*> get_neighbor_list() {
        vector<QuadTreeNode*> neighbor_list;

        if (parent == nullptr) return neighbor_list;

        // Add siblings
        for (int i = 0; i < 4; ++i) {
            QuadTreeNode* sibling = parent->children[i];
            if (sibling != this) {
                neighbor_list.push_back(sibling);
            }
        }

        // Get neighbors of parent
        vector<QuadTreeNode*> parent_neighbors = parent->get_neighbor_list();
        for (QuadTreeNode* neighbor : parent_neighbors) {
            if (neighbor->children[0] != nullptr) {
                for (int i = 0; i < 4; ++i) {
                    QuadTreeNode* child = neighbor->children[i];
                    if (child != nullptr && is_adjacent(child)) {
                        neighbor_list.push_back(child);
                    }
                }
            } else if (is_adjacent(neighbor)) {
                neighbor_list.push_back(neighbor);
            }
        }

        return neighbor_list;
    }

    // Check if a node is adjacent to this node
    bool is_adjacent(QuadTreeNode* node) {
        double dx = abs(node->x_center - x_center);
        double dy = abs(node->y_center - y_center);
        double max_size = 0.5 * ((x_max - x_min) + (node->x_max - node->x_min));
        return dx <= max_size && dy <= max_size;
    }

    // Check if a node is well-separated
    bool is_well_separated(QuadTreeNode* node) {
        // Nodes are well-separated if they are not neighbors
        return !is_adjacent(node);
    }
};

// Global variables
vector<Particle> particles;
QuadTreeNode* root = nullptr;

// Function prototypes
void init_simulation(int num_particles);
void compute_forces();
void update_particles(double dt);
void render();
void render_particles();
void render_quadtree();

// OpenGL error callback
void error_callback(int error, const char* description) {
    cerr << "Error: " << description << endl;
}

// GLFW key callback
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // Close window on escape key
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW." << endl;
        return -1;
    }

    glfwSetErrorCallback(error_callback);

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "FMM Visualization", NULL, NULL);
    if (!window) {
        cerr << "Failed to create GLFW window." << endl;
        glfwTerminate();
        return -1;
    }

    // Set key callback
    glfwSetKeyCallback(window, key_callback);

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW (required for modern OpenGL)
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW." << endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Set up the viewport
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // Set up orthographic projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);

    // Set the clear color to black
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Disable depth testing
    glDisable(GL_DEPTH_TEST);

    // Enable point smoothing
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

    
    init_simulation(num_particles);
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);
        glLoadIdentity(); // Reset modelview matrix

        // Compute forces
        compute_forces();

        // Update particles
        update_particles(dt);

        // Render simulation
        render();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
        // usleep(500000);
    }

    // Clean up
    delete root;
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void init_simulation(int num_particles) {
    // Define simulation domain
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    // Define cluster centers
    vector<pair<double, double>> cluster_centers = {
        {0.2, 0.8},
        {0.6, 0.4},
        {0.25, 0.75},
        {0.75, 0.75}
    };

    // Standard deviation for particle spread around each cluster center
    double cluster_spread = 0.05;

    // Create particles
    particles.clear();
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, cluster_spread);

    // particles.emplace_back(0.5, 0.5, 200.0);
    // particles.emplace_back(0.75, 0.25, 100000.0);
    for (int i = 0; i < num_particles; ++i) {
        // Assign particle to a random cluster
        auto& center = cluster_centers[i % cluster_centers.size()];

        // Generate position around the cluster center
        double x = center.first + distribution(generator);
        double y = center.second + distribution(generator);

        // Ensure particles remain within the domain bounds
        x = max(x_min, min(x_max, x));
        y = max(y_min, min(y_max, y));

        // Assign mass (fixed or randomized)
        double mass = 1.0;

        particles.emplace_back(x, y, mass);
    }
}

void compute_forces() {
    // Reset forces
    for (Particle& p : particles) {
        p.fx = 0.0;
        p.fy = 0.0;
    }

    // Build quadtree
    if (root != nullptr) delete root;
    root = new QuadTreeNode(0.0, 1.0, 0.0, 1.0, 0, nullptr);
    for (Particle& p : particles) {
        root->insert_particle(&p);
    }

    // Compute multipole expansions
    root->compute_multipole();

    // Set root's local expansion to zero
    for (int k = 0; k <= P; ++k) {
        root->local_coeffs[k] = complex<double>(0.0, 0.0);
    }

    // Compute local expansions
    root->compute_local_expansion();

    // Evaluate forces
    root->evaluate();
}

void update_particles(double dt) {
    // Update particle positions (simple Euler integration)
    for (Particle& p : particles) {
        // Update velocities
        p.vx += p.fx * dt;
        p.vy += p.fy * dt;

        // Update positions
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        // Keep particles within bounds and reflect velocities
        if (p.x < 0.0) {
            p.x = 0.0;
            p.vx *= -0.5; // Dampen on collision
        }
        if (p.x > 1.0) {
            p.x = 1.0;
            p.vx *= -0.5;
        }
        if (p.y < 0.0) {
            p.y = 0.0;
            p.vy *= -0.5;
        }
        if (p.y > 1.0) {
            p.y = 1.0;
            p.vy *= -0.5;
        }
        // if(p.mass==2000.0) continue;
        // if (pow(p.x - particles[0].x, 2)+pow(p.y - particles[0].y, 2) < 0.0001) {
        //     p.vx *= -0*p.vx; // Dampen on collision
        //     p.vy *= -0*p.vy; // Dampen on collision
        // }
        p.history.emplace_back(p.x, p.y);
        if (p.history.size() > MAX_HISTORY) {
            p.history.pop_front();
        }
        
    }
}

void render() {
    // Render particles
    render_particles();

    // Render quadtree
    render_quadtree();
}

void render_particles() {
    glPointSize(5.0f); // Increased point size
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow particles for better visibility
    for (Particle& p : particles) {
        glVertex2f(p.x, p.y);
    }
    glEnd();
    
    for (Particle& p : particles) {
        glColor3f(0.0f, 1.0f, 0.0f); // Green paths
        glBegin(GL_LINE_STRIP);
        for (const auto& position : p.history) {
            glVertex2f(position.first, position.second);
        }
        glEnd();
    }
}

void render_quadtree() {
    if (root != nullptr) {
        root->render_cells();
    }
}
