"""Contains all the functions to generate the model objects: model box, model compartments and the model particles (MP ans SPM)"""

import math
import copy

from globalConstants import *
from compartment_classes import *
from particulate_classes import *


class Box:
    """Class box generates one object box representing the unit world in the case of the UTOPIA parameterization that can contain 17 compartments and that can have connexions to other boxes (for example if conecting several UTOPIA boxes to give spatial resolution to a global model)"""

    description = "Generic Box class"

    def __init__(
        self,
        Bname,
        Bdepth_m=None,
        Blength_m=None,
        Bwidth_m=None,
        Bvolume_m3=None,
        Bconexions=None,
    ):
        # Assign attributes to self (instance attributes). Those set up as None are optional attributes
        self.Bname = Bname  # name of the box
        self.Bdepth_m = Bdepth_m  # depth of the box
        self.Blength_m = Blength_m  # length of the box
        self.Bwidth_m = Bwidth_m  # width of the box
        self.Bvolume_m3 = Bvolume_m3  # volume of the box
        self.compartments = []  # list of compartments in the box
        self.Bconexions = Bconexions  # conexions to other model boxes

    def __repr__(self):
        return (
            "{"
            + self.Bname
            + ", "
            + str(self.Bdepth_m)
            + ", "
            + str(self.Blength_m)
            + ", "
            + str(self.Bwidth_m)
            + "}"
        )

    def add_compartment(self, comp):
        self.compartments.append(comp)
        comp.assign_box(self)

    def calc_Bvolume_m3(self):
        if self.Bvolume_m3 is None:
            if any(
                attr is None for attr in [self.Bdepth_m, self.Blength_m, self.Bwidth_m]
            ):
                print(
                    "Missing parameters needded to calculate Box volume --> calculating based on compartments volume"
                )
                if len(self.compartments) == 0:
                    print(
                        "No compartments assigned to this model box --> use add_compartment(comp)"
                    )
                else:
                    vol = []
                    for c in range(len(self.compartments)):
                        if self.compartments[c].Cvolume_m3 is None:
                            print(
                                "Volume of compartment "
                                + self.compartments[c].Cname
                                + " is missing"
                            )
                            continue
                        else:
                            vol.append(self.compartments[c].Cvolume_m3)
                    self.Bvolume_m3 = sum(vol)
            else:
                self.Bvolume_m3 = self.Bdepth_m * self.Blength_m * self.Bwidth_m
                # print("Box volume: " + str(self.Bvolume_m3)+" m3")
        else:
            print("Box volume already assigned: " + str(self.Bvolume_m3) + " m3")


from readinputs_from_csv import *
from create_inputsTable_UTOPIA import *


def generate_objects(
    inputs_path,
    boxName,
    MPforms_list,
    comp_input_file_name,
    comp_interactFile_name,
    mp_imputFile_name,
    spm_density_kg_m3,
    spm_radius_um,
):
    """Function for generating the UTOPIA model objects: model box, model compartments and the model particles"""
    # Boxes
    UTOPIA = Box(boxName)
    # print(f"The model box {boxName} has been created")

    modelBoxes = [UTOPIA]
    # modelBoxes=instantiateBoxes_from_csv(boxFile)
    boxNames_list = [b.Bname for b in modelBoxes]

    # Compartmets
    """Call read imput file function for compartments"""

    compartments = instantiate_compartments(inputs_path + comp_input_file_name)

    # Establish connexions between compartments defining their interaction mechanism: only listed those compartments wich will recieve particles from the define compartment. i.e. the ocean surface water compartment transports particles to the ocean mix layer through settling and to air through sea spray resuspension

    set_interactions(
        compartments, connexions_path_file=inputs_path + comp_interactFile_name
    )

    # Assign modelling code to compartments
    for c in range(len(compartments)):
        compartments[c].Ccode = c + 1

    ##Calculate compartments volume
    for c in compartments:
        c.calc_volume()

    ## Dictionary of compartments
    dict_comp = {
        item.Cname: item for item in compartments
    }  # Before the compartments association to RS...migth need to come after to also reflect the CBox connexion

    compartmentNames_list = [item.Cname for item in compartments]

    # PARTICLES

    ##Free microplastics (freeMP)

    # MP_freeParticles = instantiateParticles_from_csv(inputs_path + mp_imputFile_name)

    MP_freeParticles = instantiateParticles_from_csv(mp_imputFile_name)

    dict_size_coding = dict(
        zip(
            [p.Pname for p in MP_freeParticles],
            [p.diameter_um for p in MP_freeParticles],
        )
    )

    ###Calculate freeMP volume
    for i in MP_freeParticles:
        i.calc_volume()
        # print(f"Density of {i.Pname}: {i.Pdensity_kg_m3} kg_m3")

    ##Biofouled microplastics (biofMP)
    spm = Particulates(
        Pname="spm1",
        Pform="suspendedParticulates",
        Pcomposition="Mixed",
        Pdensity_kg_m3=spm_density_kg_m3,
        Pshape="sphere",
        PdimensionX_um=spm_radius_um,
        PdimensionY_um=0,
        PdimensionZ_um=0,
    )
    spm.calc_volume()
    # print(f"spm Volume: {spm.Pvolume_m3} m3")
    # print(f"Density of spm: {spm.Pdensity_kg_m3} kg_m3")

    MP_biofouledParticles = []
    for i in MP_freeParticles:
        MP_biofouledParticles.append(ParticulatesBF(parentMP=i, spm=spm))
    # print(
    #     f"The biofouled MP particles {[p.Pname for p in MP_biofouledParticles]} have been generated"
    # )

    ###Calculate biofMP volume
    for i in MP_biofouledParticles:
        i.calc_volume()
        # print(f"Density of {i.Pname}: {i.Pdensity_kg_m3} kg_m3")

    ##Heteroaggregated microplastics (heterMP)

    MP_heteroaggregatedParticles = []
    for i in MP_freeParticles:
        MP_heteroaggregatedParticles.append(ParticulatesSPM(parentMP=i, parentSPM=spm))
    # print(
    #     f"The heteroaggregated MP particles {[p.Pname for p in MP_heteroaggregatedParticles]} have been generated"
    # )

    ###Calculate heterMP volume
    for i in MP_heteroaggregatedParticles:
        i.calc_volume_heter(i.parentMP, spm)
        # print(f"Density of {i.Pname}: {i.Pdensity_kg_m3} kg_m3")

    ##Biofouled and Heteroaggregated microplastics (biofHeterMP)
    MP_biofHeter = []
    for i in MP_biofouledParticles:
        MP_biofHeter.append(ParticulatesSPM(parentMP=i, parentSPM=spm))
    # for i in MP_biofHeter:
    #     print(f"Density of {i.Pname}: {i.Pdensity_kg_m3} kg_m3")
    # print(
    #     f"The biofouled and heteroaggregated MP particles {[p.Pname for p in MP_biofHeter]} have been generated"
    # )

    ###Calculate biofHeterMP volume
    for i in MP_biofHeter:
        i.calc_volume_heter(i.parentMP, spm)

    particles = (
        MP_freeParticles
        + MP_biofouledParticles
        + MP_heteroaggregatedParticles
        + MP_biofHeter
    )

    particles_properties = {
        "Particle": ([p.Pname for p in particles]),
        "Radius_m": ([p.radius_m for p in particles]),
        "Volume_m3": ([p.Pvolume_m3 for p in particles]),
        "Density_kg_m3": ([p.Pdensity_kg_m3 for p in particles]),
        "Corey Shape Factor": ([p.CSF for p in particles]),
    }

    particles_df = pd.DataFrame(data=particles_properties)
    # print(particles_df)
    # particles_df.to_csv("Particles_properties_output.csv", index=False)

    # Assign compartmets to UTOPIA

    for comp in compartments:
        UTOPIA.add_compartment(
            copy.deepcopy(comp)
        )  # Check if the use of copy is correct!!

    # print(
    #     f"The compartments {[comp.Cname for comp in UTOPIA.compartments]} have been assigned to {UTOPIA.Bname } model box"
    # )

    # Estimate volume of UTOPIA box by adding volumes of the compartments addedd
    # UTOPIA.calc_Bvolume_m3() #currently volume of soil and air boxess are missing, to be added to csv file

    # Add particles to compartments
    for b in modelBoxes:
        for c in b.compartments:
            for p in particles:
                c.add_particles(copy.deepcopy(p))
        # print(f"The particles have been added to the compartments of {b.Bname}")

    # List of particle objects in the system:
    system_particle_object_list = []

    for b in modelBoxes:
        for c in b.compartments:
            for freeMP in c.particles["freeMP"]:
                system_particle_object_list.append(freeMP)
            for heterMP in c.particles["heterMP"]:
                system_particle_object_list.append(heterMP)
            for biofMP in c.particles["biofMP"]:
                system_particle_object_list.append(biofMP)
            for heterBiofMP in c.particles["heterBiofMP"]:
                system_particle_object_list.append(heterBiofMP)

    # Generate list of species names and add code name to object
    SpeciesList = generate_system_species_list(
        system_particle_object_list, MPforms_list, compartmentNames_list, boxNames_list
    )

    model_lists = dict(
        zip(
            ["compartmentNames_list", "boxNames_list", "dict_size_coding"],
            [compartmentNames_list, boxNames_list, dict_size_coding],
        )
    )

    return (
        system_particle_object_list,
        SpeciesList,
        spm,
        dict_comp,
        model_lists,
        particles_df,
    )


###CONTINUE HERE###
