from numba import njit

####

from mcdc.constant import COINCIDENCE_TOLERANCE, COINCIDENCE_TOLERANCE_TIME
import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng

from mcdc.transport.distribution import (
    sample_distribution_with_scale,
    sample_uniform,
    sample_tabulated,
    sample_tabulated_with_interval,
    sample_pmf,
    sample_white_direction,
    sample_isotropic_direction,
)
from mcdc.transport.linalg import direction_from_angles
from mcdc.transport.util import find_bin_with_rules


@njit
def source_particle(particle_container, seed, simulation, data):
    particle = particle_container[0]
    particle["rng_seed"] = seed

    # Sample source
    xi = rng.lcg(particle_container)
    source_cdf = mcdc_get.simulation.source_cdf_all(simulation, data)
    source_idx = find_bin_with_rules(xi, source_cdf, 0.0, False)
    source = simulation["sources"][source_idx]
    weight = 1.0

    # Position
    (x, y, z), multiplier = sample_position(particle_container, source, simulation, data)
    weight *= multiplier

    # Direction
    (ux, uy, uz), (mu, polar_interval), multiplier = sample_direction(particle_container, source, simulation, data)
    weight *= multiplier

    # Energy
    E, multiplier = sample_energy(particle_container, source, mu, polar_interval, simulation, data)
    weight *= multiplier

    # Time
    t, multiplier = sample_time(particle_container, source, simulation, data)
    weight *= multiplier

    # Motion translation
    if source["moving"]:
        dx, dy, dz = motion_translation(t, source, data)
        x += dx
        y += dy
        z += dz

    # Make and return particle
    particle["x"] = x
    particle["y"] = y
    particle["z"] = z
    particle["t"] = t
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz
    particle["E"] = E
    particle["w"] = weight
    particle["particle_type"] = source["particle_type"]


@njit
def sample_position(particle_container, source, simulation, data):
    """
    Sample particle position.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    source : object 
        Source object from simulation containing distributions.
    simulation : object
        Simulation state containing distributions.
    data : object
        Simulation data for array access.

    Returns
    -------
    (x, y, z) : Tuple[float]
        Sampled particle position.
    weight_multiplier : float
        Multiplier for biasing particle weights. Default 1.0.
    """
    weight_multiplier = 1.0
    if source["point_source"]:
        x = source["point"][0]
        y = source["point"][1]
        z = source["point"][2]
    else:
        x = sample_position_axis(
            source["uniform_x"],
            source["x"],
            source["x_pdf_ID"],
            particle_container,
            simulation,
            data,
        )
        y = sample_position_axis(
            source["uniform_y"],
            source["y"],
            source["y_pdf_ID"],
            particle_container,
            simulation,
            data,
        )
        z = sample_position_axis(
            source["uniform_z"],
            source["z"],
            source["z_pdf_ID"],
            particle_container,
            simulation,
            data,
        )
    return (x, y, z), weight_multiplier


@njit
def sample_direction(particle_container, source, simulation, data):
    """
    Sample particle direction.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    source : object 
        Source object from simulation containing distributions.
    simulation : object
        Simulation state containing distributions.
    data : object
        Simulation data for array access.

    Returns
    -------
    (ux, uy, uz) : Tuple[float]
        Sampled particle direction.
    (mu, polar_interval) : float
    weight_multiplier : float
        Multiplier for biasing particle weights. Default 1.0.
    """
    weight_multiplier = 1.0
    mu = 0.0
    polar_interval = 0
    if source["isotropic_direction"]:
        ux, uy, uz = sample_isotropic_direction(particle_container)
    elif source["white_direction"]:
        rx = source["direction"][0]
        ry = source["direction"][1]
        rz = source["direction"][2]
        ux, uy, uz = sample_white_direction(rx, ry, rz, particle_container)
    elif source["mono_direction"]:
        ux = source["direction"][0]
        uy = source["direction"][1]
        uz = source["direction"][2]
    else:
        if source["uniform_polar_cosine"]:
            mu = sample_uniform(
                source["polar_cosine"][0],
                source["polar_cosine"][1],
                particle_container,
            )
        else:
            if source["energy_at_polar_cosine_active"]:
                mu, polar_interval = _sample_source_tabulated_with_interval(
                    source["polar_cosine_pdf_ID"],
                    particle_container,
                    simulation,
                    data,
                )
            else:
                mu = _sample_source_tabulated(
                    source["polar_cosine_pdf_ID"],
                    particle_container,
                    simulation,
                    data,
                )

        if source["uniform_azimuthal"]:
            azi = sample_uniform(
                source["azimuthal"][0],
                source["azimuthal"][1],
                particle_container,
            )
        else:
            azi = _sample_source_tabulated(
                source["azimuthal_pdf_ID"],
                particle_container,
                simulation,
                data,
            )

        ux, uy, uz = direction_from_angles(mu, azi, source["direction"])

    return (ux, uy, uz), (mu, polar_interval), weight_multiplier


@njit
def sample_energy(particle_container, source, mu, polar_interval, simulation, data):
    """
    Sample particle energy.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    source : object 
        Source object from simulation containing distributions.
    mu : float
        Polar cosine.
    polar_interval : float
    simulation : object
        Simulation state containing distributions.
    data : object
        Simulation data for array access.

    Returns
    -------
    E : float
        Sampled particle energy.
    weight_multiplier : float
        Multiplier for biasing particle weights. Default 1.0.
    """
    weight_multiplier= 1.0
    if source["energy_at_polar_cosine_active"]:
        if source["energy_at_polar_cosine_is_distribution"]:
            ID = source["energy_at_polar_cosine_distribution_ID"]
            distribution = simulation["distributions"][ID]
            E = sample_distribution_with_scale(
                mu,
                distribution,
                particle_container,
                simulation,
                data,
            )
        else:
            E = _interpolate_energy_at_polar_cosine(
                mu,
                polar_interval,
                source,
                simulation,
                data,
            )
    elif source["mono_energetic"]:
        E = source["energy"]
    elif source["discrete_energy"]:
        ID = source["energy_pmf_ID"]
        sub_ID = simulation["distributions"][ID]["sub_ID"]
        pmf = simulation["pmf_distributions"][sub_ID]
        E = sample_pmf(pmf, particle_container, data)
    else:
        E = _sample_source_tabulated(
            source["energy_pdf_ID"], particle_container, simulation, data
        )

    return E, weight_multiplier


@njit
def sample_time(particle_container, source, simulation, data):
    """
    Sample particle time.

    Parameters
    ----------
    particle_container : ndarray
        Container holding the particle.
    source : object 
        Source object from simulation containing distributions.
    simulation : object
        Simulation state containing distributions.
    data : object
        Simulation data for array access.

    Returns
    -------
    t : float
        Sampled particle time.
    weight_multiplier : float
        Multiplier for biasing particle weights. Default 1.0.
    """
    weight_multiplier = 1.0
    if source["discrete_time"]:
        t = source["time"]
    elif source["uniform_time"]:
        t = sample_uniform(
            source["time_range"][0], source["time_range"][1], particle_container
        )
    else:
        t = _sample_source_tabulated(
            source["time_pdf_ID"], particle_container, simulation, data
        )
    return t, weight_multiplier 


@njit
def motion_translation(t, source, data):
    """
    Determine motion translation.

    Parameters
    ----------
    t : float
        Time.
    source : object 
        Source object from simulation containing distributions.
    data : object
        Simulation data for array access.

    Returns
    -------
    (dx, dy, dz) : Tuple[float, float, float]
        Translated distances.
    """
    # Get moving interval index wrt the given time
    time_grid = mcdc_get.source.move_time_grid_all(source, data)

    tolerance = COINCIDENCE_TOLERANCE_TIME
    go_lower = False
    idx = find_bin_with_rules(t, time_grid, tolerance, go_lower)

    # Coinciding cases
    if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
        idx += 1

    # Source move translations
    trans_0 = mcdc_get.source.move_translations_vector(idx, source, data)

    # Source move velocities
    V = mcdc_get.source.move_velocities_vector(idx, source, data)

    # Source move time grid
    time_0 = mcdc_get.source.move_time_grid(idx, source, data)

    # Translate the particle
    t_local = t - time_0

    dx = trans_0[0] + V[0] * t_local
    dy = trans_0[1] + V[1] * t_local
    dz = trans_0[2] + V[2] * t_local
    return dx, dy, dz


@njit
def sample_position_axis(uniform, bounds, pdf_ID, rng_state, simulation, data):
    """Sample one independent source-position coordinate."""
    if uniform:
        return sample_uniform(bounds[0], bounds[1], rng_state)

    return _sample_source_tabulated(pdf_ID, rng_state, simulation, data)


@njit
def _sample_source_tabulated(pdf_ID, rng_state, simulation, data):
    """Sample one of a source's tabulated continuous distributions."""
    sub_ID = simulation["distributions"][pdf_ID]["sub_ID"]
    table = simulation["tabulated_distributions"][sub_ID]
    return sample_tabulated(table, rng_state, simulation, data)


@njit
def _sample_source_tabulated_with_interval(pdf_ID, rng_state, simulation, data):
    """Sample a source distribution and retain its interpolation interval."""
    sub_ID = simulation["distributions"][pdf_ID]["sub_ID"]
    table = simulation["tabulated_distributions"][sub_ID]
    return sample_tabulated_with_interval(table, rng_state, simulation, data)


@njit
def _interpolate_energy_at_polar_cosine(mu, interval, source, simulation, data):
    """Interpolate deterministic source energy in the sampled cosine interval."""
    pdf_ID = source["polar_cosine_pdf_ID"]
    sub_ID = simulation["distributions"][pdf_ID]["sub_ID"]
    table = simulation["tabulated_distributions"][sub_ID]
    pdf_data = simulation["data"][table["pdf_ID"]]
    pdf_table = simulation["table_data"][pdf_data["sub_ID"]]

    mu_0 = mcdc_get.table_data.x(interval, pdf_table, data)
    mu_1 = mcdc_get.table_data.x(interval + 1, pdf_table, data)
    energy_0 = mcdc_get.source.energy_at_polar_cosine(interval, source, data)
    energy_1 = mcdc_get.source.energy_at_polar_cosine(interval + 1, source, data)

    fraction = (mu - mu_0) / (mu_1 - mu_0)
    return energy_0 + fraction * (energy_1 - energy_0)
