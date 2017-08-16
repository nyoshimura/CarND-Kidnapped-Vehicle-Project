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
#include <math.h>
//#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>

#include "particle_filter.h"

using namespace std;
std::default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
	num_particles = 1000;
	// define standard deviation
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	// define distribution
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	// initialize all particles
	for(int i = 0; i < num_particles; i++)
	{
		Particle particle = {i, dist_x(generator), dist_y(generator), dist_theta(generator), 1};
		particles.push_back(particle);
		// all weights to 1
		float w = 1;
		weights.push_back(w);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// define standard deviation
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	double yaw_delta = delta_t * yaw_rate;

	for(auto &particle : particles)
	{
		// add measurements to each particle
		double x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_delta) - sin(particle.theta));
		double y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_delta));
		double theta = particle.theta + yaw_delta;
		// define distribution
		normal_distribution<double> dist_x(x, std_x);
		normal_distribution<double> dist_y(y, std_y);
		normal_distribution<double> dist_theta(theta, std_theta);
		// add random gaussian noise
		particle.x = dist_x(generator);
		particle.y = dist_y(generator);
		particle.theta = dist_theta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution
	for(auto &particle : particles)
	{
		// set param of each particle
		double particle_x = particle.x;
		double particle_y = particle.y;
		double particle_theta = particle.theta;
		// standard deviation of landmark
		double sigma_x = std_landmark[0];
		double sigma_y = std_landmark[1];
		// initialize weight
		double weight = 1;

		// iterate over observations
		for(auto &observation : observations)
		{
			double x = observation.x;
			double y = observation.y;
			// transform observation from vehicle to map coordinates
			double map_x = particle_x + x * cos(particle_theta) - y * sin(particle_theta);
			double map_y = particle_y + x * sin(particle_theta) + y * cos(particle_theta);
			// find closest landmark
			double min_dist = numeric_limits<double>::max();
			Map::single_landmark_s closest_landmark;

			for(auto& landmark : map_landmarks.landmark_list)
			{
				double dist_x = map_x - landmark.x_f;
				double dist_y = map_y - landmark.y_f;
				double dist = dist_x * dist_x + dist_y * dist_y;
				if(dist < min_dist)
				{
					min_dist = dist;
					closest_landmark = landmark;
				}
			}
			// calculate weight using multivariate gaussian distribution
			double dx = closest_landmark.x_f - map_x;
			double dy = closest_landmark.y_f - map_y;
			double expo = -1.0 * ( (dx * dx) / (2.0 * sigma_x * sigma_x) + (dy * dy) / (2.0 * sigma_y * sigma_y));
			double num = exp(expo);
			double denom = 2 * M_PI * sigma_x * sigma_y;
			weight *= num / denom;
		}
		particle.weight = weight;
		weights[particle.id] = weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	double max_weight = *max_element(begin(weights), end(weights));
	uniform_real_distribution<> rand_beta(0.0, 2.0 * max_weight);

	// define new vectors
	vector<double> new_weights;
	vector<Particle> new_parts;

	// randomly pick an initial point in the wheel
	uniform_int_distribution<int> wheel_initial_distribution(0, weights.size() - 1);
	int index = wheel_initial_distribution(generator);
	assert(index < weights.size());

	// sample new values
	double beta = 0;
	for(int i = 0; i < weights.size(); i++)
	{
		beta += rand_beta(generator); // skip ahead

		// update index until weights becomes smaller than beta
		while(weights[index] < beta)
		{
			beta -= weights[index]; // decrease beta
			index = (index + 1) % weights.size(); // increase index
		}

		Particle new_part = {i, particles[index].x, particles[index].y, particles[index].theta, weights[index]};
		new_weights.push_back(weights[index]);
		new_parts.push_back(new_part);
	}
	weights = new_weights;
	particles = new_parts;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
