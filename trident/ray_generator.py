"""
SpectrumGenerator class and member functions.

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, Trident Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

from trident.light_ray import \
    LightRay
from yt.loaders import \
    load
from trident.config import \
    ion_table_filepath
from trident.line_database import \
    LineDatabase, \
    uniquify
from trident.roman import \
    from_roman
from yt.data_objects.static_output import \
    Dataset
from trident.ion_balance import \
    atomic_number
from trident.meshless_voronoi_ray import \
    MeshlessVoronoiRayTracer
from trident.meshless_ray_io import \
    load_meshless_ray, \
    write_meshless_ray_hdf5, \
    write_meshless_ray_catalog_hdf5
from yt.utilities.logger import \
    ytLogger as mylog
from yt.utilities.physical_constants import \
    speed_of_light_cgs
import numpy as np

def make_simple_ray(dataset_file, start_position, end_position,
                    lines=None, ftype="gas", fields=None,
                    solution_filename=None, data_filename=None,
                    trajectory=None, redshift=None, field_parameters=None,
                    setup_function=None, load_kwargs=None,
                    line_database=None, ionization_table=None,
                    fail_empty=True):
    """
    Create a yt LightRay object for a single dataset (eg CGM).  This is a
    wrapper function around yt's LightRay interface to reduce some of the
    complexity there.

    A simple ray is a straight line passing through a single dataset
    where each gas cell intersected by the line is sampled for the desired
    fields and stored.  Several additional fields are created and stored
    including ``dl`` to represent the path length in space
    for each element in the ray, ``v_los`` to represent the line of
    sight velocity along the ray, and ``redshift``, ``redshift_dopp``, and
    ``redshift_eff`` to represent the cosmological redshift, doppler redshift
    and effective redshift (combined doppler and cosmological) for each
    element of the ray.

    A simple ray is typically specified by its start and end positions in the
    dataset volume.  Because a simple ray only probes a single output, it
    lacks foreground absorbers between the observer at z=0 and the redshift
    of the dataset that one would naturally encounter.  Thus it is usually
    only appropriate for studying the circumgalactic medium rather than
    the intergalactic medium.

    This function can accept a yt dataset already loaded in memory,
    or it can load a dataset if you pass it the dataset's filename and
    optionally any load_kwargs or setup_function necessary to load/process it
    properly before generating the LightRay object.

    The :lines: keyword can be set to automatically add all fields to the
    resulting ray necessary for later use with the SpectrumGenerator class.
    If the necessary fields do not exist for your line of choice, they will
    be added to your dataset before adding them to the ray.

    **Parameters**

    :dataset_file: string or yt Dataset object

        Either a yt dataset or the filename of a dataset on disk.  If you are
        passing it a filename, consider usage of the ``load_kwargs`` and
        ``setup_function`` kwargs.

    :start_position, end_position: list of floats or YTArray object

        The coordinates of the starting and ending position of the desired
        ray.  If providing a raw list, coordinates are assumed to be in
        code length units, but if providing a YTArray, any units can be
        specified.

    :lines: list of strings, optional

        List of strings that determine which fields will be added to the ray
        to support line deposition to an absorption line spectrum.  List can
        include things like "C", "O VI", or "Mg II ####", where #### would be
        the integer wavelength value of the desired line.  If set to 'all',
        includes all possible ions from H to Zn. :lines: can be used
        in conjunction with :fields: as they will not override each other.
        Default: None

    :ftype: string, optional

        This is now deprecated and unnecessary.
        Default: "gas"

    :fields: list of strings, optional

        The list of which fields to store in the output LightRay.
        See :lines: keyword for additional functionality that will add fields
        necessary for creating absorption line spectra for certain line
        features.
        Default: None

    :solution_filename: string, optional

        Output filename of text file containing trajectory of LightRay
        through the dataset.
        Default: None

    :data_filename: string, optional

        Output filename for ray data stored as an HDF5 file.  Note that
        at present, you *must* save a ray to disk in order for it to be
        returned by this function.  If set to None, defaults to 'ray.h5'.
        Default: None

    :trajectory: list of floats, optional

        The (r, theta, phi) direction of the LightRay.  Use either end_position
        or trajectory, but not both.
        Default: None

    :redshift: float, optional

        Sets the highest cosmological redshift of the ray.  By default, it will
        use the cosmological redshift of the dataset, if set, and if not set,
        it will use a redshift of 0.
        Default: None

    :field_parameters: optional, dict
        Used to set field parameters in light rays. For example,
        if the 'bulk_velocity' field parameter is set, the relative
        velocities used to calculate peculiar velocity will be adjusted
        accordingly.
        Default: None.

    :setup_function: function, optional

        A function that will be called on the dataset as it is loaded but
        before the LightRay is generated.  Very useful for adding derived
        fields and other manipulations of the dataset prior to LightRay
        creation.
        Default: None

    :load_kwargs: dict, optional

        Dictionary of kwargs to be passed to the yt "load" function prior to
        creating the LightRay.  Very useful for many frontends like Gadget,
        Tipsy, etc. for passing in "bounding_box", "unit_base", etc.
        Default: None

    :line_database: string, optional

        For use with the :lines: keyword. If you want to limit the available
        ion fields to be added to those available in a particular subset,
        you can use a :class:`~trident.LineDatabase`.  This means when you
        set :lines:='all', it will only use those ions present in the
        corresponding LineDatabase.  If :LineDatabase: is set to None,
        and :lines:='all', it will add every ion of every element up to Zinc.
        Default: None

    :ionization_table: string, optional

        For use with the :lines: keyword.  Path to an appropriately formatted
        HDF5 table that can be used to compute the ion fraction as a function
        of density, temperature, metallicity, and redshift.  When set to None,
        it uses the table specified in ~/.trident/config
        Default: None

    :fail_empty: optional, bool

        If True, Trident will fail when it tries to create an empty Ray
        that does not pass through any valud fluid elements. When
        False, it will merely return a warning.
        Default: True

    **Example**

    Generate a simple ray passing from the lower left corner to the upper
    right corner through some Gizmo dataset:

    >>> import trident
    >>> import yt
    >>> ds = yt.load('path/to/dataset')
    >>> ray = trident.make_simple_ray(ds,
    ... start_position=ds.domain_left_edge, end_position=ds.domain_right_edge,
    ... lines=['H', 'O', 'Mg II'])
    """
    if load_kwargs is None:
        load_kwargs = {}
    if fields is None:
        fields = []
    if data_filename is None:
        data_filename = 'ray.h5'

    if isinstance(dataset_file, str):
        ds = load(dataset_file, **load_kwargs)
    elif isinstance(dataset_file, Dataset):
        ds = dataset_file

    lr = LightRay(ds, load_kwargs=load_kwargs)

    if ionization_table is None:
        ionization_table = ion_table_filepath

    # Include some default fields in the ray to assure it's processed correctly.

    fields = _add_default_fields(ds, fields)

    # If 'lines' kwarg is set, we need to get all the fields required to
    # create the desired absorption lines in the grid format, since grid-based
    # fields are what are directly probed by the LightRay object.

    # We first determine what fields are necessary for the desired lines, and
    # inspect the dataset to see if they already exist.  If so, we add them
    # to the field list for the ray.  If not, we have to create them.

    if lines is not None:

        ion_list = _determine_ions_from_lines(line_database, lines)
        fields = _determine_fields_from_ions(ds, ion_list, fields)

    # To assure there are no fields that are double specified or that collide
    # based on being specified as "density" as well as ("gas", "density"),
    # we will just assume that all non-tuple fields requested are ftype "gas".
    for i in range(len(fields)):
        if isinstance(fields[i], str):
            fields[i] = ('gas', fields[i])
    fields = uniquify(fields)

    return lr.make_light_ray(start_position=start_position,
                             end_position=end_position,
                             trajectory=trajectory,
                             fields=fields,
                             setup_function=setup_function,
                             solution_filename=solution_filename,
                             data_filename=data_filename,
                             field_parameters=field_parameters,
                             redshift=redshift,
                             fail_empty=fail_empty)

def make_meshless_voronoi_ray(
        dataset_file,
        start_position,
        end_position=None,
        direction=None,
        length=None,
        lines=None,
        ftype="gas",
        fields=None,
        solution_filename=None,
        data_filename=None,
        redshift=None,
        field_parameters=None,
        setup_function=None,
        load_kwargs=None,
        line_database=None,
        ionization_table=None,
        position_field=("gas", "coordinates"),
        velocity_field=None,
        periodic=False,
        extra_ray_fields=True,
        store_meshless_metadata=True,
        fail_empty=True,
        **meshless_kwargs):
    """
    Create a Trident-compatible ray using meshless Voronoi geometry.

    This function mirrors :func:`make_simple_ray` for a single loaded dataset,
    but replaces yt's native ray traversal with a SALSA-style meshless Voronoi
    walk over gas/cell generating-site positions.  The saved ray is a yt data
    dataset tagged as ``yt_light_ray`` so it can be consumed by Trident's
    existing :class:`~trident.SpectrumGenerator`.
    """
    if load_kwargs is None:
        load_kwargs = {}
    if field_parameters is None:
        field_parameters = {}
    if fields is None:
        fields = []
    else:
        fields = list(fields)
    if data_filename is None:
        data_filename = 'ray.h5'

    if isinstance(dataset_file, str):
        ds = load(dataset_file, **load_kwargs)
    elif isinstance(dataset_file, Dataset):
        ds = dataset_file
    else:
        raise RuntimeError("dataset_file must be a filename or yt Dataset.")

    if setup_function is not None:
        setup_function(ds)

    if ionization_table is None:
        ionization_table = ion_table_filepath

    fields = _add_default_fields(ds, fields)
    if lines is not None:
        ion_list = _determine_ions_from_lines(line_database, lines)
        fields = _determine_fields_from_ions(ds, ion_list, fields)

    for i in range(len(fields)):
        if isinstance(fields[i], str):
            fields[i] = ('gas', fields[i])
    fields = uniquify(fields)

    ad = ds.all_data()
    for key, val in field_parameters.items():
        ad.set_field_parameter(key, val)

    position_field, positions = _resolve_meshless_position_field(
        ds, ad, position_field
    )
    positions_code = positions.to('code_length')

    start_code = _meshless_position_to_code(ds, start_position)
    ray_kwargs = {"start_position": start_code}
    if end_position is not None:
        ray_kwargs["end_position"] = _meshless_position_to_code(ds, end_position)
    else:
        if direction is None or length is None:
            raise ValueError("Provide end_position or both direction and length.")
        ray_kwargs["direction"] = np.asarray(direction, dtype=np.float64)
        ray_kwargs["length"] = _meshless_length_to_code(ds, length)

    box_size = meshless_kwargs.pop("box_size", None)
    if periodic:
        if box_size is None:
            box_size = ds.domain_width.to('code_length').d
    else:
        box_size = None

    tracer = MeshlessVoronoiRayTracer(
        positions_code.d,
        box_size=box_size,
        **meshless_kwargs
    )
    meshless_ray = tracer.trace_ray(**ray_kwargs)
    ray_indices = meshless_ray.indices

    velocity = _resolve_meshless_velocity(ds, ad, velocity_field, len(positions_code))
    velocity = velocity[ray_indices]
    bulk_velocity = field_parameters.get("bulk_velocity", None)
    if bulk_velocity is not None:
        if hasattr(bulk_velocity, "to"):
            bulk_velocity = bulk_velocity.to('cm/s')
        else:
            bulk_velocity = ds.arr(bulk_velocity, 'cm/s')
        velocity = velocity - bulk_velocity

    # Match yt LightRay's convention: line_of_sight points from the ray end
    # back toward the ray start, i.e. start - end.
    line_of_sight = ds.arr(-meshless_ray.direction, "")
    velocity_los = (velocity * line_of_sight).sum(axis=1).to('cm/s')

    dl = ds.arr(meshless_ray.dl, 'code_length').in_cgs()
    cumulative_dl = ds.arr(meshless_ray.cumulative_dl, 'code_length').in_cgs()
    l = cumulative_dl - 0.5 * dl

    redshift_arr = _meshless_redshift_array(
        ds, redshift, l, ds.quan(meshless_ray.length, 'code_length').in_cgs()
    )
    velocity_mag = np.sqrt((velocity * velocity).sum(axis=1)).to('cm/s')
    beta2 = (velocity_mag / speed_of_light_cgs).to("").d ** 2
    beta2 = np.clip(beta2, 0.0, 1.0 - np.finfo(np.float64).eps)
    redshift_dopp = (
        (1.0 + (velocity_los / speed_of_light_cgs).to("").d) /
        np.sqrt(1.0 - beta2)
    ) - 1.0
    redshift_eff = ((1.0 + redshift_arr) * (1.0 + redshift_dopp)) - 1.0

    data = {
        ('gas', 'dl'): dl,
        ('gas', 'l'): l,
        ('gas', 'redshift'): redshift_arr,
        ('gas', 'redshift_dopp'): redshift_dopp,
        ('gas', 'redshift_eff'): redshift_eff,
        ('gas', 'velocity_los'): velocity_los,
        ('gas', 'v_los'): velocity_los,
    }

    for field in fields:
        if field in data:
            continue
        data[field] = _sample_meshless_field(ad, field, ray_indices)

    l_code = meshless_ray.cumulative_dl - 0.5 * meshless_ray.dl
    ray_positions_code = (
        meshless_ray.start_position[None, :] +
        l_code[:, None] * meshless_ray.direction[None, :]
    )
    if meshless_ray.periodic and meshless_ray.box_size is not None:
        ray_positions_code = np.mod(ray_positions_code, meshless_ray.box_size)
    ray_positions = ds.arr(ray_positions_code, 'code_length').in_cgs()
    generator_positions = positions[ray_indices].to('code_length').in_cgs()
    data[('gas', 'x')] = ray_positions[:, 0]
    data[('gas', 'y')] = ray_positions[:, 1]
    data[('gas', 'z')] = ray_positions[:, 2]

    if extra_ray_fields:
        data[('gas', 'meshless_cell_index')] = ray_indices.astype(np.int64)
        data[('gas', 'cumulative_dl')] = cumulative_dl
        data[('gas', 'meshless_generator_x')] = generator_positions[:, 0]
        data[('gas', 'meshless_generator_y')] = generator_positions[:, 1]
        data[('gas', 'meshless_generator_z')] = generator_positions[:, 2]
        data[('gas', 'relative_velocity_x')] = velocity[:, 0].to('cm/s')
        data[('gas', 'relative_velocity_y')] = velocity[:, 1].to('cm/s')
        data[('gas', 'relative_velocity_z')] = velocity[:, 2].to('cm/s')

    extra_attrs = {}
    if store_meshless_metadata:
        extra_attrs.update(_meshless_metadata_attrs(ds, meshless_ray, position_field))

    if solution_filename is not None:
        _write_meshless_solution(solution_filename, meshless_ray)

    write_meshless_ray_hdf5(
        ds,
        data_filename,
        data,
        extra_attrs=extra_attrs,
        fail_empty=fail_empty,
    )
    return load_meshless_ray(data_filename)

def make_meshless_voronoi_ray_catalog(
        dataset_file,
        starts,
        ends=None,
        directions=None,
        lengths=None,
        lines=None,
        fields=None,
        output_filename=None,
        output_dir=None,
        one_file_per_ray=False,
        parallel="none",
        n_jobs=1,
        chunksize=1000,
        position_field=("gas", "coordinates"),
        velocity_field=None,
        periodic=False,
        redshift=None,
        field_parameters=None,
        setup_function=None,
        load_kwargs=None,
        line_database=None,
        ionization_table=None,
        diagnostics=True,
        overwrite=False,
        fail_empty=True,
        **meshless_kwargs):
    """Trace and optionally write a catalog of meshless Voronoi sightlines.

    This high-level batch API loads the dataset once, builds one meshless
    nearest-neighbor tree, traces all supplied rays, samples requested fields
    with flattened cell indices, and writes a compact ragged HDF5 catalog when
    ``output_filename`` is supplied.

    Existing single-ray behavior is unchanged; use ``make_meshless_voronoi_ray``
    when a Trident/SpectrumGenerator-ready single ray file is needed directly.
    """
    if load_kwargs is None:
        load_kwargs = {}
    if field_parameters is None:
        field_parameters = {}
    if fields is None:
        fields = []
    else:
        fields = list(fields)

    if isinstance(dataset_file, str):
        ds = load(dataset_file, **load_kwargs)
    elif isinstance(dataset_file, Dataset):
        ds = dataset_file
    else:
        raise RuntimeError("dataset_file must be a filename or yt Dataset.")

    if setup_function is not None:
        setup_function(ds)
    if ionization_table is None:
        ionization_table = ion_table_filepath

    fields = _add_default_fields(ds, fields)
    if lines is not None:
        ion_list = _determine_ions_from_lines(line_database, lines)
        fields = _determine_fields_from_ions(ds, ion_list, fields)
    for i in range(len(fields)):
        if isinstance(fields[i], str):
            fields[i] = ('gas', fields[i])
    fields = uniquify(fields)

    ad = ds.all_data()
    for key, val in field_parameters.items():
        ad.set_field_parameter(key, val)

    position_field, positions = _resolve_meshless_position_field(ds, ad, position_field)
    positions_code = positions.to('code_length')
    starts_code = _meshless_positions_to_code_array(ds, starts)
    if ends is not None:
        ends_code = _meshless_positions_to_code_array(ds, ends)
        directions_code = None
        lengths_code = None
    else:
        ends_code = None
        directions_code = np.asarray(directions, dtype=np.float64)
        lengths_code = _meshless_lengths_to_code_array(ds, lengths)

    box_size = meshless_kwargs.pop("box_size", None)
    if periodic:
        if box_size is None:
            box_size = ds.domain_width.to('code_length').d
    else:
        box_size = None

    tracer = MeshlessVoronoiRayTracer(
        positions_code.d,
        box_size=box_size,
        **meshless_kwargs
    )
    batch = tracer.trace_rays(
        starts_code,
        end_positions=ends_code,
        directions=directions_code,
        lengths=lengths_code,
        parallel=parallel,
        n_jobs=n_jobs,
        chunksize=chunksize,
        return_format="ragged",
    )

    data = _meshless_catalog_flat_fields(
        ds, ad, batch, fields, velocity_field, field_parameters, redshift,
        len(positions_code), positions
    )

    metadata = {
        "meshless_position_field": str(position_field),
        "meshless_extra_fields_version": "1",
        "n_rays": int(batch.n_rays),
        "n_segments": int(len(batch.indices)),
        "diagnostics": bool(diagnostics),
    }
    if hasattr(ds, "unique_identifier"):
        metadata["source_unique_identifier"] = str(ds.unique_identifier)
    if hasattr(ds, "parameter_filename"):
        metadata["source_parameter_filename"] = str(ds.parameter_filename)

    if output_filename is not None:
        write_meshless_ray_catalog_hdf5(
            output_filename,
            batch,
            fields=data,
            metadata=metadata,
            overwrite=overwrite,
        )

    if one_file_per_ray:
        if output_dir is None:
            raise ValueError("output_dir is required when one_file_per_ray=True.")
        _write_catalog_individual_rays(
            ds, batch, data, output_dir, fail_empty=fail_empty, overwrite=overwrite
        )

    if output_filename is not None:
        return output_filename
    return batch

def make_compound_ray(parameter_filename, simulation_type,
                      near_redshift, far_redshift,
                      lines=None, ftype='gas', fields=None,
                      solution_filename=None, data_filename=None,
                      use_minimum_datasets=True, max_box_fraction=1.0,
                      deltaz_min=0.0, minimum_coherent_box_fraction=0.0,
                      find_outputs=False, seed=None,
                      setup_function=None, load_kwargs=None,
                      line_database=None, ionization_table=None,
                      field_parameters = None,
                      fail_empty=True):
    """
    Create a yt LightRay object for multiple consecutive datasets (eg IGM).
    This is a wrapper function around yt's LightRay interface to reduce some
    of the complexity there.

    .. note::

        The compound ray functionality has only been implemented for the
        Enzo and Gadget/Gizmo codes.  If you would like to help us implement
        this functionality for your simulation code, please contact us
        about this on the mailing list.

    A compound ray is a series of straight lines passing through multiple
    consecutive outputs from a single cosmological simulation to approximate
    a continuous line of sight to high redshift.

    Because a single continuous ray traversing a simulated volume can only
    cover a small range in redshift space (e.g. 100 Mpc only covers the
    redshift range from z=0 to z=0.023), the compound ray passes rays through
    multiple consecutive outputs from the same simulation to approximate the
    path of a single line of sight to high redshift.  By probing all of the
    foreground material out to any given redshift, the compound ray is
    appropriate for studies of the intergalactic medium and circumgalactic
    medium.

    By default, it selects a random starting location and trajectory in
    each dataset it traverses, to assure that the same cosmological structures
    are not being probed multiple times from the same direction.  In doing
    this, the ray becomes discontinuous across each dataset.

    The compound ray requires the parameter_filename of the simulation run.
    This is *not* the dataset filename from a single output, but the parameter
    file that was used to run the simulation itself.  It is in this parameter
    file that the output frequency, simulation volume, and cosmological
    parameters are described to assure full redshift coverage can be achieved
    for a compound ray.  It also requires the simulation_type of the simulation.

    Unlike the simple ray, which is specified by its start and end positions
    in the dataset volume, the compound ray requires the near_redshift and
    far_redshift to determine which datasets to use to get full coverage
    in redshift space as the ray propagates from near_redshift to far_redshift.

    Like the simple ray produced by :class:`~trident.make_simple_ray`,
    each gas cell intersected by the LightRay is sampled for the desired
    fields and stored.  Several additional fields are created and stored
    including ``dl`` to represent the path length in space
    for each element in the ray, ``v_los`` to represent the line of
    sight velocity along the ray, and ``redshift``, ``redshift_dopp``, and
    ``redshift_eff`` to represent the cosmological redshift, doppler redshift
    and effective redshift (combined doppler and cosmological) for each
    element of the ray.

    The :lines: keyword can be set to automatically add all fields to the
    resulting ray necessary for later use with the SpectrumGenerator class.

    **Parameters**

    :parameter_filename: string

        The simulation parameter file *not* the dataset filename

    :simulation_type: string

        The simulation type of the parameter file.  At present, this
        functionality only works with "Enzo" and "Gadget" yt frontends.

    :near_redshift, far_redshift: floats

        The near and far redshift bounds of the LightRay through the
        simulation datasets.

    :lines: list of strings, optional

        List of strings that determine which fields will be added to the ray
        to support line deposition to an absorption line spectrum.  List can
        include things like "C", "O VI", or "Mg II ####", where #### would be
        the integer wavelength value of the desired line.  If set to 'all',
        includes all possible ions from H to Zn. :lines: can be used
        in conjunction with :fields: as they will not override each other.
        Default: None

    :ftype: string, optional

        This is now deprecated and unnecessary.
        Default: "gas"

    :fields: list of strings, optional

        The list of which fields to store in the output LightRay.
        See :lines: keyword for additional functionality that will add fields
        necessary for creating absorption line spectra for certain line
        features.
        Default: None

    :solution_filename: string, optional

        Output filename of text file containing trajectory of LightRay
        through the dataset.
        Default: None

    :data_filename: string, optional

        Output filename for ray data stored as an HDF5 file.  Note that
        at present, you *must* save a ray to disk in order for it to be
        returned by this function.  If set to None, defaults to 'ray.h5'.
        Default: None

    :use_minimum_datasets: bool, optional

        Use the minimum number of datasets to make the ray continuous
        through the supplied datasets from the near_redshift to the
        far_redshift.  If false, the LightRay solution will contain as many
        datasets as possible to enable the light ray to traverse the
        desired redshift interval.
        Default: True

    :max_box_fraction: float, optional

        The maximum length a light ray segment can be in order to span the
        redshift interval from one dataset to another in units of the domain
        size.  Values larger than 1.0 will result in LightRays crossing the
        domain of a given dataset more than once, which is generally undesired.
        Zoom-in simulations can use a value equal to the length of the
        high-resolution region so as to limit ray segments to that size.  If
        the high-resolution region is not cubical, the smallest size should b
        used.
        Default: 1.0 (the size of the box)

    :deltaz_min: float, optional

        The minimum delta-redshift value between consecutive datasets used
        in the LightRay solution.
        Default: 0.0

    :minimum_coherent_box_fraction: float, optional

        When use_minimum_datasets is set to False, this parameter specifies
        the fraction of the total box width to be traversed before
        rerandomizing the ray location and trajectory.
        Default: 0.0

    :find_outputs: optional, bool

        Whether or not to search for datasets in the current
        directory. This is useful if the number of existing datasets is
        different than what would be predicted by the simulation parameter file.
        Default: False.

    :seed: int, optional

        Sets the seed for the random number generator used to determine the
        location and trajectory of the LightRay as it traverses the
        simulation datasets.  For consistent results between LightRays,
        use the same seed value.
        Default: None

    :setup_function: function, optional

        A function that will be called on the dataset as it is loaded but
        before the LightRay is generated.  Very useful for adding derived
        fields and other manipulations of the dataset prior to LightRay
        creation.
        Default: None

    :load_kwargs: dict, optional

        Dictionary of kwargs to be passed to the yt "load" function prior to
        creating the LightRay.  Very useful for many frontends like Gadget,
        Tipsy, etc. for passing in "bounding_box", "unit_base", etc.
        Default: None

    :line_database: string, optional

        For use with the :lines: keyword. If you want to limit the available
        ion fields to be added to those available in a particular subset,
        you can use a :class:`~trident.LineDatabase`.  This means when you
        set :lines:='all', it will only use those ions present in the
        corresponding LineDatabase.  If :LineDatabase: is set to None,
        and :lines:='all', it will add every ion of every element up to Zinc.
        Default: None

    :ionization_table: string, optional

        For use with the :lines: keyword.  Path to an appropriately formatted
        HDF5 table that can be used to compute the ion fraction as a function
        of density, temperature, metallicity, and redshift.  When set to None,
        it uses the table specified in ~/.trident/config
        Default: None

    :field_parameters: optional, dict
        Used to set field parameters in light rays. For example,
        if the 'bulk_velocity' field parameter is set, the relative
        velocities used to calculate peculiar velocity will be adjusted
        accordingly.
        Default: None.

    :fail_empty: optional, bool

        If True, Trident will fail when it tries to create an empty Ray
        that does not pass through any valud fluid elements. When
        False, it will merely return a warning.
        Default: True

    **Example**

    Generate a compound ray passing from the redshift 0 to redshift 0.05
    through a multi-output enzo simulation.

    >>> import trident
    >>> fn = 'path/to/simulation/parameter/file'
    >>> ray = trident.make_compound_ray(fn, simulation_type='Enzo',
    ... near_redshift=0.0, far_redshift=0.05, lines=['H', 'O', 'Mg II'])

    Generate a compound ray passing from the redshift 0 to redshift 0.05
    through a multi-output gadget simulation.

    >>> import trident
    >>> fn = 'path/to/simulation/parameter/file'
    >>> ray = trident.make_compound_ray(fn, simulation_type='Gadget',
    ... near_redshift=0.0, far_redshift=0.05, lines=['H', 'O', 'Mg II'])
    """
    if load_kwargs is None:
        load_kwargs = {}
    if fields is None:
        fields = []
    if data_filename is None:
        data_filename = 'ray.h5'

    lr = LightRay(parameter_filename,
                  simulation_type=simulation_type,
                  near_redshift=near_redshift,
                  far_redshift=far_redshift,
                  find_outputs=find_outputs,
                  use_minimum_datasets=use_minimum_datasets,
                  max_box_fraction=max_box_fraction,
                  deltaz_min=deltaz_min,
                  minimum_coherent_box_fraction=minimum_coherent_box_fraction,
                  load_kwargs=load_kwargs)

    if ionization_table is None:
        ionization_table = ion_table_filepath

    # We use the final dataset from the light ray solution in order to test it for
    # what fields are present, etc.  This all assumes that the fields present
    # in this output will be present in ALL outputs.  Hopefully this is true,
    # because testing each dataset is going to be slow and a pain.

    ds = load(lr.light_ray_solution[-1]['filename'])

    # Include some default fields in the ray to assure it's processed correctly.

    fields = _add_default_fields(ds, fields)

    # If 'lines' kwarg is set, we need to get all the fields required to
    # create the desired absorption lines in the grid format, since grid-based
    # fields are what are directly probed by the LightRay object.

    # We first determine what fields are necessary for the desired lines, and
    # inspect the dataset to see if they already exist.  If so, we add them
    # to the field list for the ray or add the necessary fields that can
    # generate them on the ray.

    if lines is not None:

        ion_list = _determine_ions_from_lines(line_database, lines)
        fields = _determine_fields_from_ions(ds, ion_list, fields)

    # To assure there are no fields that are double specified or that collide
    # based on being specified as "density" as well as ("gas", "density"),
    # we will just assume that all non-tuple fields requested are ftype "gas".
    for i in range(len(fields)):
        if isinstance(fields[i], str):
            fields[i] = ('gas', fields[i])
    fields = uniquify(fields)

    return lr.make_light_ray(seed=seed,
                             fields=fields,
                             setup_function=setup_function,
                             solution_filename=solution_filename,
                             data_filename=data_filename,
                             redshift=None, njobs=-1,
                             field_parameters = field_parameters,
                             fail_empty=fail_empty)

def _meshless_position_to_code(ds, position):
    if hasattr(position, "to"):
        return position.to('code_length').d
    return ds.arr(position, 'code_length').to('code_length').d

def _meshless_length_to_code(ds, length):
    if hasattr(length, "to"):
        return float(length.to('code_length').d)
    return float(ds.quan(length, 'code_length').to('code_length').d)

def _meshless_positions_to_code_array(ds, positions):
    if hasattr(positions, "to"):
        arr = positions.to('code_length').d
    else:
        arr = ds.arr(positions, 'code_length').to('code_length').d
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("positions must have shape (M, 3).")
    return arr

def _meshless_lengths_to_code_array(ds, lengths):
    if hasattr(lengths, "to"):
        arr = lengths.to('code_length').d
    else:
        arr = ds.arr(lengths, 'code_length').to('code_length').d
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("lengths must have shape (M,).")
    return arr

def _meshless_redshift(ds, redshift):
    if redshift is not None:
        return float(redshift)
    if hasattr(ds, "current_redshift"):
        try:
            return float(ds.current_redshift)
        except TypeError:
            pass
    return 0.0

def _meshless_redshift_array(ds, redshift, l, total_length):
    z_start = _meshless_redshift(ds, redshift)
    if len(l) == 0:
        return np.array([], dtype=np.float64)
    if not getattr(ds, "cosmological_simulation", False):
        return np.full(len(l), z_start, dtype=np.float64)
    try:
        segment_length = total_length.in_units("Mpccm / h")
        z_next = z_start - LightRay(ds)._deltaz_forward(z_start, segment_length)
        fraction = (l / total_length).to("").d
        return z_start - fraction * (z_start - float(z_next))
    except Exception:
        return np.full(len(l), z_start, dtype=np.float64)

def _resolve_meshless_position_field(ds, ad, position_field):
    if _meshless_component_field_spec(position_field):
        components = [ad[field].to('code_length') for field in position_field]
        positions = ds.arr(np.column_stack([component.d for component in components]),
                           'code_length')
        return position_field, positions

    candidates = [position_field]
    for fallback in [("gas", "coordinates"), ("PartType0", "Coordinates")]:
        if fallback not in candidates:
            candidates.append(fallback)

    errors = []
    for candidate in candidates:
        try:
            positions = ad[candidate]
        except Exception as exc:
            errors.append("%s: %s" % (candidate, exc))
            continue
        if len(getattr(positions, "shape", ())) != 2 or positions.shape[1] != 3:
            errors.append("%s: expected shape (N, 3), got %s" %
                          (candidate, getattr(positions, "shape", None)))
            continue
        if candidate != position_field:
            mylog.warning(
                "Using fallback meshless position field %s instead of %s.",
                candidate, position_field
            )
        if candidate[0] == "PartType0":
            mylog.warning(
                "Using raw PartType0 coordinates as Voronoi generating sites."
            )
        return candidate, positions

    raise RuntimeError(
        "Could not resolve a meshless position field. Tried: %s" %
        "; ".join(errors)
    )

def _resolve_meshless_velocity(ds, ad, velocity_field, n_positions):
    if velocity_field is not None:
        if _meshless_component_field_spec(velocity_field):
            components = [ad[field].to('cm/s') for field in velocity_field]
            return ds.arr(np.column_stack([component.d for component in components]),
                          'cm/s')
        try:
            velocity = ad[velocity_field]
        except Exception as exc:
            raise RuntimeError(
                "Could not load requested velocity_field %s: %s" %
                (velocity_field, exc)
            )
        if len(getattr(velocity, "shape", ())) != 2 or velocity.shape[1] != 3:
            raise RuntimeError(
                "velocity_field %s must have shape (N, 3)." % (velocity_field,)
            )
        return velocity.to('cm/s')

    component_sets = [
        (("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")),
        (("gas", "relative_velocity_x"), ("gas", "relative_velocity_y"),
         ("gas", "relative_velocity_z")),
    ]
    for fields in component_sets:
        try:
            components = [ad[field].to('cm/s') for field in fields]
        except Exception:
            continue
        return ds.arr(np.column_stack([component.d for component in components]),
                      'cm/s')

    for candidate in [("gas", "velocity"), ("PartType0", "Velocities")]:
        try:
            velocity = ad[candidate]
        except Exception:
            continue
        if len(getattr(velocity, "shape", ())) == 2 and velocity.shape[1] == 3:
            if candidate[0] == "PartType0":
                mylog.warning("Using raw PartType0 velocities for velocity_los.")
            return velocity.to('cm/s')

    mylog.warning(
        "No gas velocity field found for meshless ray; using zero velocity_los."
    )
    return ds.arr(np.zeros((n_positions, 3), dtype=np.float64), 'cm/s')

def _meshless_component_field_spec(field_spec):
    return (
        isinstance(field_spec, (list, tuple)) and
        len(field_spec) == 3 and
        all(isinstance(field, tuple) for field in field_spec)
    )

def _sample_meshless_field(ad, field, indices):
    try:
        values = ad[field]
    except Exception as exc:
        raise RuntimeError(
            "Required meshless ray field %s is not available: %s" %
            (field, exc)
        )
    return values[indices]

def _sample_meshless_field_flat(ad, field, indices):
    try:
        values = ad[field]
    except Exception as exc:
        raise RuntimeError(
            "Required meshless ray field %s is not available: %s" %
            (field, exc)
        )
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    return values[unique_indices][inverse]

def _meshless_catalog_flat_fields(
        ds, ad, batch, fields, velocity_field, field_parameters, redshift,
        n_positions, positions):
    ray_ids = np.repeat(np.arange(batch.n_rays), batch.n_segments)
    ray_indices = batch.indices
    velocity_all = _resolve_meshless_velocity(ds, ad, velocity_field, n_positions)
    velocity = velocity_all[ray_indices]
    bulk_velocity = field_parameters.get("bulk_velocity", None)
    if bulk_velocity is not None:
        if hasattr(bulk_velocity, "to"):
            bulk_velocity = bulk_velocity.to('cm/s')
        else:
            bulk_velocity = ds.arr(bulk_velocity, 'cm/s')
        velocity = velocity - bulk_velocity

    directions_flat = ds.arr(-batch.directions[ray_ids], "")
    velocity_los = (velocity * directions_flat).sum(axis=1).to('cm/s')
    dl = ds.arr(batch.dl, 'code_length').in_cgs()
    cumulative_dl = ds.arr(batch.cumulative_dl, 'code_length').in_cgs()
    l = cumulative_dl - 0.5 * dl
    l_code = batch.cumulative_dl - 0.5 * batch.dl

    redshift_arr = np.empty(len(ray_indices), dtype=np.float64)
    for ray_id in range(batch.n_rays):
        start = batch.ray_offsets[ray_id]
        end = batch.ray_offsets[ray_id + 1]
        total_length = ds.quan(batch.lengths[ray_id], 'code_length').in_cgs()
        redshift_arr[start:end] = _meshless_redshift_array(
            ds, redshift, l[start:end], total_length
        )
    velocity_mag = np.sqrt((velocity * velocity).sum(axis=1)).to('cm/s')
    beta2 = (velocity_mag / speed_of_light_cgs).to("").d ** 2
    beta2 = np.clip(beta2, 0.0, 1.0 - np.finfo(np.float64).eps)
    redshift_dopp = (
        (1.0 + (velocity_los / speed_of_light_cgs).to("").d) /
        np.sqrt(1.0 - beta2)
    ) - 1.0
    redshift_eff = ((1.0 + redshift_arr) * (1.0 + redshift_dopp)) - 1.0

    ray_positions_code = (
        batch.start_positions[ray_ids] +
        l_code[:, None] * batch.directions[ray_ids]
    )
    if batch.periodic and batch.box_size is not None:
        ray_positions_code = np.mod(ray_positions_code, batch.box_size)
    ray_positions = ds.arr(ray_positions_code, 'code_length').in_cgs()
    generator_positions = positions[ray_indices].to('code_length').in_cgs()

    data = {
        ('gas', 'dl'): dl,
        ('gas', 'l'): l,
        ('gas', 'redshift'): redshift_arr,
        ('gas', 'redshift_dopp'): redshift_dopp,
        ('gas', 'redshift_eff'): redshift_eff,
        ('gas', 'velocity_los'): velocity_los,
        ('gas', 'v_los'): velocity_los,
        ('gas', 'meshless_cell_index'): ray_indices.astype(np.int64),
        ('gas', 'cumulative_dl'): cumulative_dl,
        ('gas', 'x'): ray_positions[:, 0],
        ('gas', 'y'): ray_positions[:, 1],
        ('gas', 'z'): ray_positions[:, 2],
        ('gas', 'relative_velocity_x'): velocity[:, 0].to('cm/s'),
        ('gas', 'relative_velocity_y'): velocity[:, 1].to('cm/s'),
        ('gas', 'relative_velocity_z'): velocity[:, 2].to('cm/s'),
    }
    data[('gas', 'meshless_generator_x')] = generator_positions[:, 0]
    data[('gas', 'meshless_generator_y')] = generator_positions[:, 1]
    data[('gas', 'meshless_generator_z')] = generator_positions[:, 2]

    for field in fields:
        if field in data:
            continue
        data[field] = _sample_meshless_field_flat(ad, field, ray_indices)
    return data

def _write_catalog_individual_rays(ds, batch, data, output_dir, fail_empty=True, overwrite=False):
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ray_id in range(batch.n_rays):
        start = int(batch.ray_offsets[ray_id])
        end = int(batch.ray_offsets[ray_id + 1])
        filename = output_dir / ("meshless_ray_%05d.h5" % ray_id)
        if filename.exists() and not overwrite:
            raise FileExistsError("%s exists. Pass overwrite=True." % filename)
        ray_data = {}
        for field, values in data.items():
            ray_data[field] = values[start:end]
        attrs = {}
        for key, value in batch.ray(ray_id).metadata.items():
            attrs["meshless_%s" % key] = value
        write_meshless_ray_hdf5(
            ds, str(filename), ray_data, extra_attrs=attrs, fail_empty=fail_empty
        )

def _meshless_metadata_attrs(ds, ray, position_field):
    attrs = {}
    for key, value in ray.metadata.items():
        attrs["meshless_%s" % key] = value
    attrs["meshless_position_field"] = str(position_field)
    attrs["meshless_extra_fields_version"] = "1"
    if hasattr(ds, "unique_identifier"):
        attrs["meshless_source_unique_identifier"] = str(ds.unique_identifier)
    if hasattr(ds, "parameter_filename"):
        attrs["meshless_source_parameter_filename"] = str(ds.parameter_filename)
    return attrs

def _write_meshless_solution(filename, ray):
    with open(filename, "w") as handle:
        handle.write("# Meshless Voronoi ray solution\n")
        handle.write("# start %s\n" % " ".join(map(str, ray.start_position)))
        handle.write("# end %s\n" % " ".join(map(str, ray.end_position)))
        handle.write("# direction %s\n" % " ".join(map(str, ray.direction)))
        handle.write("# length %s\n" % ray.length)
        handle.write("# index dl cumulative_dl\n")
        for index, dl, cumulative in zip(ray.indices, ray.dl, ray.cumulative_dl):
            handle.write("%d %.17e %.17e\n" % (index, dl, cumulative))

def _determine_ions_from_lines(line_database, lines):
    """
    Figure out what ions are necessary to produce the desired lines
    """
    if line_database is not None:
        line_database = LineDatabase(line_database)
        ion_list = line_database.parse_subset_to_ions(lines)
    else:
        ion_list = []
        if lines == 'all' or lines == ['all']:
            for k,v in atomic_number.items():
                for j in range(v+1):
                    ion_list.append((k, j+1))
        else:
            for line in lines:
                linen = line.split()
                if len(linen) >= 2:
                    ion_list.append((linen[0], from_roman(linen[1])))
                elif len(linen) == 1:
                    num_states = atomic_number[linen[0]]
                    for j in range(num_states+1):
                        ion_list.append((linen[0], j+1))
                else:
                    raise RuntimeError("Cannot add a blank ion.")

    return uniquify(ion_list)

def _determine_fields_from_ions(ds, ion_list, fields):
    """
    Figure out what fields need to be added based on the ions present.

    Check if the number_density fields for these ions exist, and if so, add
    them to field list. If not, leave them off, as they'll be generated
    on the fly by SpectrumGenerator as long as we include the 'density',
    'temperature', and appropriate 'metallicity' fields.
    """
    for ion in ion_list:
        atom = ion[0].capitalize()
        ion_state = ion[1]
        nuclei_field = "%s_nuclei_mass_density" % atom
        metallicity_field = "%s_metallicity" % atom
        field = "%s_p%d_number_density" % (atom, ion_state-1)

        # Check to see if the ion field exists.  If so, add
        # it to the ray.  If not, then append the density and the appropriate
        # metal field so one can create the ion field on the fly on the
        # ray itself.
        if ("gas", field) not in ds.derived_field_list:
            fields.append(('gas', 'density'))
            if ('gas', metallicity_field) in ds.derived_field_list:
                fields.append(('gas', metallicity_field))
            elif ('gas', nuclei_field) in ds.derived_field_list:
                fields.append(('gas', nuclei_field))
            elif atom != 'H':
                fields.append(('gas', 'metallicity'))
            else:
                # Don't need metallicity field if we're just looking
                # at hydrogen
                pass
        else:
            fields.append(("gas", field))

    return fields

def _add_default_fields(ds, fields):
    """
    Add some default fields to rays to assure they can be processed correctly.
    """
    if ("gas", "temperature") in ds.derived_field_list:
        fields.append(("gas", 'temperature'))

    # H_nuclei_density should be added if possible to assure that the _log_nH
    # field, which is used as "density" in the ion_balance interpolation to
    # produce ion fields, is calculated as accurately as possible.
    if ('gas', 'H_nuclei_density') in ds.derived_field_list:
        fields.append(('gas', 'H_nuclei_density'))

    return fields
