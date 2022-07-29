import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

num_particles = 1000
particles = np.zeros((num_particles, 6))
particles[:, 0] = np.random.uniform(-150, 150, len(particles))
particles[:, 1] = np.random.uniform(-150, 150, len(particles))
# particles[:, 2] = np.random.uniform(-np.pi, np.pi, len(particles))
# particles[:, 3] = np.random.uniform(0, 5, len(particles))

# Create simulated measurement and hydrophone state
measurement = np.array([50, 0])
measurement_cov = np.array([[100, 0], [0, 100]])
hydrophone_state = np.array([0, 0, 0, 0])

def measurement_step(particles, measurement, measurement_cov, hydrophone_state):
    # Unpack hydrophone state
    x, y, theta, v = hydrophone_state
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # Unpack the particles
    tag_x, tag_y, tag_theta, tag_v, _, _ = particles.T
    tag_vx = tag_v * np.cos(tag_theta)
    tag_vy = tag_v * np.sin(tag_theta)

    # Compute predicted measurement given the state
    x_diff = x - tag_x
    y_diff = y - tag_y
    r_pred = np.sqrt(np.square(x_diff) + np.square(y_diff))
    r_dot_pred = (x_diff * (vx - tag_vx) + y_diff * (vy - tag_vy)) / r_pred

    measurement_pred = np.column_stack([r_pred, r_dot_pred])
    dist = multivariate_normal(measurement, measurement_cov)
    weight = dist.pdf(measurement_pred)

    resampled_particles = np.copy(particles)
    resampled_particles[:, -1] = weight
    return resampled_particles

def resample(particles):
    min_weight = 1e-60
    particles[:, -1] = np.maximum(particles[:, -1], min_weight)
    w_tot = np.sum(particles[:, -1])  # sum up all the particles
    probs = particles[:, -1] = particles[:, -1] / w_tot
    indices = np.random.choice(len(particles), len(particles), p=probs)
    return particles[indices]

resampled_particles = resample(measurement_step(particles, measurement, measurement_cov, hydrophone_state))

resampled_particles[:, 0] += np.random.uniform(-1, 1)
resampled_particles[:, 1] += np.random.uniform(-1, 1)

plt.scatter(particles[:, 0], particles[:, 1])
plt.scatter(resampled_particles[:, 0], resampled_particles[:, 1])
plt.xlim((-150, 150))
plt.ylim((-150, 150))
plt.gca().set_aspect('equal', 'box')
plt.show()