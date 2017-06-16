/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // set number of particles
  num_particles = 500;

  // set random number generator
  std::default_random_engine gen;

  // Standard deviations for x, y, and psi
  double std_x, std_y, std_theta; 

  // Set standard deviations for x, y, and theta
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];  

  // Generate a normal (Gaussian) distribution for x, y and theta.
  std::normal_distribution<double> dist_x(x, std_x);
  std::normal_distribution<double> dist_y(y, std_y);
  std::normal_distribution<double> dist_theta(theta, std_theta);

  // set particle weights to 1
  weights.clear();
  weights.resize(num_particles, 1.0);

  // initialise particles
  particles.clear();
  particles.resize(num_particles);

  for (int i = 0; i < num_particles; ++i) {

    particles[i].id = 0;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;

  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Standard deviations for x, y, and theta
  // Create delta_yaw variable
  double std_x, std_y, std_theta, delta_yaw; 

  // Set standard deviations for x, y, and theta
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];
  delta_yaw = yaw_rate * delta_t;

  // set random number generator
  std::random_device rd; 
  std::default_random_engine gen(rd());

  for (int i = 0; i < num_particles; ++i) {

    double yaw = particles[i].theta;

    // Update new particle location
    if (fabs(yaw_rate) > 0.001) {
      // update x and y position and yaw rate
      particles[i].x += velocity / yaw_rate * (sin(yaw + delta_yaw) - sin(yaw));
      particles[i].y += velocity / yaw_rate * (cos(yaw) - cos(yaw + delta_yaw));
      particles[i].theta += delta_yaw;
    } else {
      // update x and y position, yaw stays the same
      particles[i].x += velocity * delta_t * cos(yaw);
      particles[i].y += velocity * delta_t * sin(yaw);
    }

    // Generate a normal (Gaussian) distribution for x, y and theta.
    std::normal_distribution<double> dist_x(particles[i].x, std_x);
    std::normal_distribution<double> dist_y(particles[i].y, std_y);
    std::normal_distribution<double> dist_theta(particles[i].theta, std_theta);

    // Update new particle location with added noise
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);

  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.


  /*********************************************
  2 enclose for loop
  1. Outer loop iterates through all predicted landmarks
  2. Inner loops finds out which observation is associated with the landmark
  **********************************************/

  // Iterate through all nearby predicted landmarks
  for (size_t i = 0; i < predicted.size(); ++i) {

    // set closest landmark, closest_dist, to max double
    double closest_dist = std::numeric_limits<double>::max();
    LandmarkObs *closest_observation = NULL;
    
    // Iterate through all observations to identify closest_dist observation
    for (size_t j = 0; j < observations.size(); ++j) {

      // Continue iteration if this observation has already been assigned to a landmark
      if (observations[j].id > 0){
        continue;
      }

      double d = dist(observations[j].x, observations[j].y, predicted[i].x, predicted[i].y);
      
      // Remember each observation that is closer to the landmark
      if (d < closest_dist) {
        closest_dist = d;
        closest_observation = &observations[j];
      }
      
    }

    // If closest_observation isn't NULL, allocated id with predicted landmark id
    if (closest_observation != NULL) {
      closest_observation->id = predicted[i].id;
    }
     
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html
  //   https://en.wikipedia.org/wiki/Rotation_matrix


  // extract sigma x,y to local variables
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  // Iterate through each particle
  for (int i = 0; i < num_particles; ++i) {

    // Observation of landmarks is in car perspective and coordinates. Therefore based on each particle
    // location and orientation, we convert the observation to actual pseudo map coordinates for each
    // particle

    // Make a copy of observations to observations_map
    std::vector<LandmarkObs> observations_map(observations);

    // Convert observations from car coordinate to map coordinate
    for (size_t t = 0; t < observations_map.size(); ++t){

      //replace observation_map with map coordinate x, y and set id value to invalid 0
      double vx = observations_map[t].x * cos(particles[i].theta) - observations_map[t].y * sin(particles[i].theta) + particles[i].x;
      double vy = observations_map[t].x * sin(particles[i].theta) + observations_map[t].y * cos(particles[i].theta) + particles[i].y;
      
      observations_map[t].x = vx;
      observations_map[t].y = vy;
      observations_map[t].id = 0; // set all observation id to invalid 0

    }

    // create predicted for storage of all relevant landmarks within car location
    std::vector<LandmarkObs> predicted;

    // iterate through all landmarks to find landmarks within car location
    for (size_t t = 0; t < map_landmarks.landmark_list.size(); ++t) {
      
      // Taking car location and landmark location on map, identify landmark as relevant if 
      // distance between car and landmark is < sensor range
      if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[t].x_f, map_landmarks.landmark_list[t].y_f) < sensor_range) {
        predicted.push_back({map_landmarks.landmark_list[t].id_i, map_landmarks.landmark_list[t].x_f, map_landmarks.landmark_list[t].y_f});
      }
      
    }

    // Implement nearest neighbour by performing landmark or data association 
    dataAssociation(predicted, observations_map);

    // calculate multi variate gaussian weight
    double particle_weight = 1;

    // Initialize c1 variable used to calculate multi variate gaussian weight
    // Implemented here for efficiency instead of iterating through the loops
    double c1 = 1.0 / (2.0 * M_PI * sigma_x * sigma_y);

    // Iterate through all observations_map that has id that is related to landmark id
    for (size_t t = 0; t < observations_map.size(); ++t) {

      // Skip observation is id is 0. i.e. non allocated landmark
      if(observations_map[t].id == 0){
        continue;
      }

      // Iterate through all predicted landmarks
      for(size_t j = 0; j < predicted.size(); ++j){

        // if predicted landmark id == observation id, calculate multi variate gaussian weight
        if(observations_map[t].id == predicted[j].id){
          // bi-variate gaussian weight
          double mu_x = predicted[j].x;
          double mu_y = predicted[j].y;
          double x = observations_map[t].x;
          double y = observations_map[t].y;

          // double c1 = 1.0 / (2.0 * M_PI * sigma_x * sigma_y); Implemented above instead of here for efficiency
          double c2 = pow(x - mu_x, 2) / pow(sigma_x, 2);
          double c3 = pow(y - mu_y, 2) / pow(sigma_y, 2);
          double weight = c1 * exp(-0.5 * (c2 + c3));

          if (weight < .0001)
            weight = .0001;

          particle_weight *= weight;

          //break once multi variate gaussian weight calculated for predicted landmark found 
          break;
        }

      }
    }

    // update particle weight with new weight
    particles[i].weight = particle_weight;

  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // intialise resampling wheel
  double beta = 0.0;

  // set random number generator
  std::random_device rd; 
  std::default_random_engine gen(rd());
  std::discrete_distribution<int> dis(0, num_particles -1);
  
  // Initialize starting index
  int index = dis(gen);

  // Initialize resampled_particles
  std::vector<Particle> resampled_particles;
  resampled_particles.reserve(num_particles);

  // find max weight and create particle weights vectors
  double max_weight = 0.0;
  weights.clear();
  weights.reserve(num_particles);
  for (size_t i = 0; i < particles.size(); ++i) {
    if (particles[i].weight > max_weight)
      max_weight = particles[i].weight;
    weights.push_back(particles[i].weight);
  }

  // create random number generator for two times max weight
  std::uniform_real_distribution<double> uniform_real(0, 2.0 * max_weight);

  // resample
  for (size_t i = 0; i < particles.size(); ++i) {
    
    beta += uniform_real(gen);
    
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  // update particles
  particles = resampled_particles;


}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}
