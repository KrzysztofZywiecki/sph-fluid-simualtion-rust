use crate::fluid_simulation::particle::Particle;
use crate::fluid_simulation::particle_dynamics_manager::ParticleDynamicsManager;
use crate::fluid_simulation::smoothed_interaction::SmoothedInteraction;
use crate::fluid_simulation::external_attractor::ExternalAttractor;
use crate::fluid_simulation::collision_manager::CollisionManager;
use crate::fluid_simulation::cell_manager::CellManager;
use piston::ReleaseEvent;
use piston::UpdateArgs;
use piston::Button;
use piston::Key;
use piston::MouseButton;
use piston::Event;
use crate::piston::PressEvent;
use vector2d::Vector2D;
use rand::Rng;
use rayon::prelude::*;

pub struct FluidSimulationApp {
  pub particles: Vec<Particle>,
  pub accelerations: Vec<Vector2D<f32>>,
  pub previous_accelerations: Vec<Vector2D<f32>>,
  pub local_densities: Vec<f32>,
  dynamics_manager: ParticleDynamicsManager,
  smoothed_interaction: SmoothedInteraction,
  external_attractor: ExternalAttractor,
  collision_manager: CollisionManager,
  cell_manager: CellManager
}

impl FluidSimulationApp {

  pub fn new(box_dimensions: [i32; 2]) -> Self {
      let mut rng = rand::thread_rng();
      let particle_count = 15000;
      let delta_time = 1.0/30.0;
      let pressure_multiplier: f32 = 90000.0;
      let target_density: f32 = 0.00003;
      let smoothing_radius: f32 = 14.0;
      let viscosity: f32 = 0.008;
      let particles: Vec<Particle> = (0..particle_count).map(
        |index| 
        Particle::new(
          index, 
          Vector2D::new(
            rng.gen_range(0.0..(300 as f32)), 
            rng.gen_range(0.0..(box_dimensions[1] as f32))
          )
        )
      ).collect();
      let local_densities: Vec<_> = particles.iter().map(|_| 0.001).collect();
      let accelerations: Vec<_> = particles.iter().map(|_| Vector2D { x: 0.0, y: 0.0 }).collect();
      let previous_accelerations: Vec<_> = accelerations.clone();
      FluidSimulationApp {
          particles,
          local_densities: local_densities,
          accelerations: accelerations,
          previous_accelerations: previous_accelerations,
          dynamics_manager: ParticleDynamicsManager::new(true, delta_time),
          smoothed_interaction: SmoothedInteraction::new(pressure_multiplier, target_density, smoothing_radius, viscosity),
          external_attractor: ExternalAttractor::new(),
          collision_manager: CollisionManager::new(box_dimensions),
          cell_manager: CellManager::new(particle_count as i32, box_dimensions, smoothing_radius)
      }
  }

  pub fn update(&mut self, _args: &UpdateArgs) {
    self.particles.par_iter_mut().enumerate().for_each(|(index, particle)| {
      self.dynamics_manager.update_position(particle, &self.accelerations[index]);
      self.collision_manager.apply_boundary_conditions(particle);
    });
    self.cell_manager.update(&mut self.particles);

    self.local_densities.par_iter_mut().enumerate().for_each(|(index, density)| {
      let particle = &self.particles[index];
      let adjacente_particles: Vec<(Particle, usize)> = self.cell_manager.get_adjancet_particles(particle.clone(), &self.particles);
      let adjecent_particles: Vec<_> = adjacente_particles.iter().map(|(p, _)| p).collect();
      *density = self.smoothed_interaction.calculate_density(particle, &adjecent_particles);
    });

    self.accelerations.par_iter_mut().enumerate().for_each(|(index, acc)| {
      let particle = &self.particles[index];
      let adjacente_particles: Vec<_> = self.cell_manager.get_adjancet_particles(particle.clone(), &self.particles);
      let adjecent_particles: Vec<_> = adjacente_particles.iter().map(|(p, index)| (p, self.local_densities[*index])).collect();
      let mut acceleration = self.smoothed_interaction.calculate_acceleration_due_to_pressure(particle, &adjecent_particles, self.local_densities[index]);
      acceleration += self.smoothed_interaction.calculate_viscosity(particle, &adjecent_particles, self.local_densities[index]);
      acceleration += self.external_attractor.get_external_attraction_acceleration(particle, self.local_densities[index]);
      *acc = acceleration;
    });

    for i in 0..self.particles.len() {
      self.dynamics_manager.update_velocity(&mut self.particles[i], &self.accelerations[i], &mut self.previous_accelerations[i]);
    }
  }

  pub fn handle_event(&mut self, event: Event) {
    if let Some(Button::Keyboard(Key::G)) = event.press_args() {
        self.dynamics_manager.toggle_gravity();
    }
    if let Some(Button::Keyboard(Key::D)) = event.press_args() {
      self.collision_manager.break_dam();
    }
    if let Some(Button::Mouse(MouseButton::Left)) = event.press_args() {
      self.external_attractor.activate(Vector2D::new(400.0 as f32, 100.0 as f32));
    }
    if let Some(Button::Mouse(MouseButton::Left)) = event.release_args() {
      self.external_attractor.active = false;
    }
  }



}
