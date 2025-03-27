import math
import pandas as pd
import os
import numpy as np
from globalConstants import *
from helpers import generate_fsd_matrix


def discorporation(particle, model):
    # Process by wich the particle looses is corporeal ("particle") form (eq to degradation) though degradation into dissolved organics?. It is considered an elimination process .
    # t_half_deg is provided as input in days and is converted to seconds. It refers to the degradation half-life of free MPs in the biggest size fraction and in the surface water compartments.
    # List of asumptions
    # The degradation rate of MPs is size, compartment and aggregation state dependent.
    # 1) Compartment: The degradation rates are slower in the deeper water compartments as well as in the soil and sediment compartments as described by the factors provided in the dict below.

    # 2) Aggregation state: The degradation rates are slower in the heteroaggregated particles (10x) but faster when biofouled (2x faster) as described by the factors provided in the dict below. Degradation in Air is also considered slower (to be revisited).

    # 3) Size: The degradation rate is scaled by the surface area to volume ratio, so that smaller particles degrade faster. In this case scale the taken degradation rate of the 50um particles since we use the data from Pfohl et al. 2022 (degradation rate of 6.3 x 10-6 for particles of TPU-ether arom in the size range between 50-200um. We asume this value as discorporation rate for the 50 um MP plastics in free form)

    MP_form_factors = {"freeMP": 1, "heterMP": 10, "biofMP": 0.5, "heterBiofMP": 5}
    compartment_factors = {
        "Ocean_Surface_Water": 1,
        "Ocean_Mixed_Water": 10,
        "Ocean_Column_Water": 10,
        "Coast_Surface_Water": 1,
        "Coast_Column_Water": 10,
        "Surface_Freshwater": 1,
        "Bulk_Freshwater": 10,
        "Sediment_Freshwater": 100,
        "Sediment_Ocean": 100,
        "Sediment_Coast": 100,
        "Beaches_Soil_Surface": 10,
        "Beaches_Deep_Soil": 100,
        "Background_Soil_Surface": 10,
        "Background_Soil": 100,
        "Impacted_Soil_Surface": 10,
        "Impacted_Soil": 100,
        "Air": 1000,
    }
    MP_size_deg_factors = (50**2) / (particle.diameter_um**2)

    # degradation half-life of MPs used as input is in days
    t_half_d = (
        model.t_half_deg_free
        * compartment_factors[particle.Pcompartment.Cname]
        * MP_form_factors[particle.Pform]
        * MP_size_deg_factors
    )
    # degradation rate constant
    k_deg = math.log(2) / (t_half_d * 24 * 60 * 60)

    return k_deg


def fragmentation(particle, model):
    # Process by wich a particle breaks into fragments of smaller sizes. The fragmentation rate (frag_rate) is estimated from the fragmentation half-life of the biggest size fraction of free MPs in the surface water compartments (tfrag_gen_d) and scaled by size, MP form and compartment factors.

    # List of asumptions
    # 1) Compartments:
    #   -Fragmentation does not take place in the Air compartment (to be revisited).
    #   -Fragmentation in the lower water compartments and in the surface of the soil takes 10 times more time than in the surface water.
    #   -Fragmentation in the sediment compartments and deeper soil take 100 times more time than in the surface water compartments.

    # 2) Aggregation state: fragmentation of biofouled particles takes double the time than for Free particles and for heteroaggregated particles it takes 100 times more.

    # 3) Size: Bigger particles fragment faster than smaller particles. The fragmentation rate is scaled by the particle diameter.

    MP_form_factors = {"freeMP": 1, "heterMP": 100, "biofMP": 2, "heterBiofMP": 200}
    compartment_factors = {
        "Ocean_Surface_Water": 1,
        "Ocean_Mixed_Water": 10,
        "Ocean_Column_Water": 10,
        "Coast_Surface_Water": 1,
        "Coast_Column_Water": 10,
        "Surface_Freshwater": 1,
        "Bulk_Freshwater": 10,
        "Sediment_Freshwater": 100,
        "Sediment_Ocean": 100,
        "Sediment_Coast": 100,
        "Beaches_Soil_Surface": 10,
        "Beaches_Deep_Soil": 100,
        "Background_Soil_Surface": 10,
        "Background_Soil": 100,
        "Impacted_Soil_Surface": 10,
        "Impacted_Soil": 100,
        "Air": 0,
    }

    # fragmentation rate in seconds
    frag_rate = (
        ((1 / (float(model.t_frag_gen_FreeSurfaceWater))) / 24 * 60 * 60)
        * MP_form_factors[particle.Pform]
        * compartment_factors[particle.Pcompartment.Cname]
        * (float(particle.diameter_um) / model.big_bin_diameter_um)
    )

    # The distribution of mass is expressed via the fragment size distribution matrix fsd (https://microplastics-cluster.github.io/fragment-mnp/advanced-usage/fragment-size-distribution.html) that is estimated from the fragmentation style of the plastic type (FI).
    # In this matrix the smallest size fraction is in the first possition and we consider no fragmentation for this size class
    fsd = generate_fsd_matrix(model.FI)
    size_positions = {chr(i): i - ord("a") for i in range(ord("a"), ord("e") + 1)}

    k_frag = frag_rate * fsd[size_positions[particle.Pcode[0]]]

    return k_frag.tolist()


def settling(particle, model):
    # settling calculations (TO BE REVISITED: different settling regimes depending on the size bin)
    """settling can be calculated using different equations (e.g. Stokes,
    modified versions of it or others) or be taken from experimental studies
    !! currently only classical Stokes is implemented (which is probably not
    very realistic and will be updated soon !!"""

    # Depending on the compartment we should use a specific water density

    if "Freshwater" in particle.Pcompartment.Cname:
        w_den_kg_m3 = density_w_21C_kg_m3
    else:
        w_den_kg_m3 = density_seaWater_kg_m3

    settlingMethod = "Stokes"

    if settlingMethod == "Stokes":
        vSet_m_s = (
            2
            / 9
            * (float(particle.Pdensity_kg_m3) - w_den_kg_m3)
            / mu_w_21C_kg_ms
            * g_m_s2
            * (float(particle.radius_m)) ** 2
        )
    else:
        print("Error: cannot calculate settling other than Stokes yet")
        # print error message settling methods other than Stokes
        # (to be removed when other settling calculations are implemented)

    # for the water and surface water compartments:
    # settling and rising rate constants for free MP
    if vSet_m_s > 0:
        k_set = vSet_m_s / float(particle.Pcompartment.Cdepth_m)

    elif vSet_m_s < 0:
        k_set = 0

    else:
        k_set = 0

    return k_set


def rising(particle, model):
    # rising calculations (TO BE REVISITED: also consider non-stokes regimes for smaller particles??)
    """rising is calculated in the same way as settling for particles with negative
    settling velocities. It can be calculated using different equations (e.g. Stokes,
    modified versions of it or others) or be taken from experimental studies
    !! currently only classical Stokes is implemented (which is probably not
    very realistic and will be updated soon !!"""

    settlingMethod = "Stokes"

    # Rising only occus in the lower water compartments wich for UTOPIA are: ["Ocean Mixed Water",
    # "Ocean Column Water","Coast Column Water","Bulk FreshWater"]

    if particle.Pcompartment.Cname in [
        "Ocean_Mixed_Water",
        "Ocean_Column_Water",
        "Coast_Column_Water",
        "Bulk_Freshwater",
    ]:

        if "Freshwater" in particle.Pcompartment.Cname:
            w_den_kg_m3 = density_w_21C_kg_m3
        else:
            w_den_kg_m3 = density_seaWater_kg_m3

        if settlingMethod == "Stokes":
            vSet_m_s = (
                2
                / 9
                * (float(particle.Pdensity_kg_m3) - w_den_kg_m3)
                / mu_w_21C_kg_ms
                * g_m_s2
                * (float(particle.radius_m)) ** 2
            )
        else:
            print("Error: cannot calculate settling other than Stokes yet")
        # print error message settling methods other than Stokes
        # (to be removed when other settling calculations are implemented)
    else:
        vSet_m_s = 0
    # for the water and surface water compartments:
    # settling and rising rate constants for free MP
    if vSet_m_s > 0:
        k_rise = 0

    elif vSet_m_s < 0:
        k_rise = -vSet_m_s / float(particle.Pcompartment.Cdepth_m)

    else:
        k_rise = 0

    return k_rise


def heteroaggregation(particle, model):
    # process of attachment of MPs to SPM particles. The rate constant is calculated based on the collision rate constant and the attachment efficiency (alpha) and the SPM number concentration.

    # Assumptions: Heteroaggegation happens to free and biofouled particles. It is hypothesized that biofilm increases the attachment efficiency of a plastic particle, reflected in two times higher values of  for biofiouled plastic particles compared to the pristine form. We assumed there is no heteroaggregation in the sediment or any soil compartment and neither in air (this is already reflected in the particle, if the particle belongs to any of these compartments there wont be heteroaggregation included as process for the particle).

    alpha_heter = {
        "freeMP": 0.01,
        "heterMP": 0,
        "biofMP": 0.02,
        "heterBiofMP": 0,
    }  # REF value: Besseling et al. 2017

    # heteroaggregation rate constants
    """heteroaggregation requires to particles to collide and interact favorably for the collision to result in attachment
    the heteroaggregation rate constants is therefore composed of two parts, 1) a collision rate constant and 2) and attachement efficiency (alpha) (representing the probability of attachement).
    For heteroaggregation a common simplifaction is the assumption that SPM concentration is not signficantly affected by the heteroaggregation process. Therefore, a pseudo first-order heteroaggregation rate constant is obtained by multiplying collision rate with alpha
    and with the SPM number concentration (REF: @AntoniaPraetorius)"""

    # first the different collision mechanisms are calculated
    k_peri = (
        (2 * k_B_J_K * float(particle.Pcompartment.T_K))
        / (3 * mu_w_21C_kg_ms)
        * (float(particle.radius_m) + model.spm.radius_m) ** 2
        / (float(particle.radius_m) * model.spm.radius_m)
    )
    # perikinetic contributions to collision rate constant (Brownian motion)

    k_ortho = (
        4
        / 3
        * float(particle.Pcompartment.G)
        * (float(particle.radius_m) + model.spm.radius_m) ** 3
    )
    # orthokinetic contributions to collision rate constant (caused by fluid motion)

    if "Freshwater" in particle.Pcompartment.Cname:
        w_den_kg_m3 = density_w_21C_kg_m3
    else:
        w_den_kg_m3 = density_seaWater_kg_m3

    MP_vSet_m_s = (
        2
        / 9
        * (float(particle.Pdensity_kg_m3) - w_den_kg_m3)
        / mu_w_21C_kg_ms
        * g_m_s2
        * (float(particle.radius_m)) ** 2
    )

    SPM_vSet_m_s = (
        2
        / 9
        * (model.spm.Pdensity_kg_m3 - w_den_kg_m3)
        / mu_w_21C_kg_ms
        * g_m_s2
        * (model.spm.radius_m) ** 2
    )
    # settling velocity. currently according to classical Stokes law. Need to include other modes and put calculation on its own, so that it can also be accessed for other processes

    k_diffSettling = (
        math.pi
        * (float(particle.radius_m) + model.spm.radius_m) ** 2
        * abs(MP_vSet_m_s - SPM_vSet_m_s)
    )

    # differential settling contributions to collision rate constant

    k_coll = k_peri + k_ortho + k_diffSettling
    # the collision rate constant

    alpha = alpha_heter[particle.Pform]

    model.spm.calc_numConc(
        concMass_mg_L=float(particle.Pcompartment.SPM_mgL), concNum_part_L=0
    )
    SPM_concNum_part_m3 = model.spm.concNum_part_m3
    k_hetAgg = float(alpha) * k_coll * SPM_concNum_part_m3
    # the pseudo first-order heteroaggregation rate constant

    return k_hetAgg


def heteroaggregate_breackup(particle, model):
    # process of breakup of heteroaggregates. The rate constant is calculated based on the collision rate constant and the attachment efficiency (alpha) and the SPM number concentration.
    """Assumption: the breack-up of heteroaggregates is 10E8 times slower than the formation of heteroaggregates (THIS HAS TO BE REVIISITED)"""
    # data is limited on aggregate breakup, but this process is likely more relvant for larger aggregates
    #!! 10E8 of k_hetAgg is just a placeholder,  needs to be refined
    # possibly using a size dependent function !!

    # Kbreackup is calculated based on Kheter of the free and biofouled MPs

    alpha_heter = {
        "freeMP": 0.01,
        "heterMP": 0,
        "biofMP": 0.02,
        "heterBiofMP": 0,
    }  # REF value: Besseling et al. 2017

    # first the different collision mechanisms are calculated

    k_peri = (
        (2 * k_B_J_K * float(particle.Pcompartment.T_K))
        / (3 * mu_w_21C_kg_ms)
        * (float(particle.radius_m) + model.spm.radius_m) ** 2
        / (float(particle.radius_m) * model.spm.radius_m)
    )
    # perikinetic contributions to collision rate constant (Brownian motion)

    k_ortho = (
        4
        / 3
        * float(particle.Pcompartment.G)
        * (float(particle.radius_m) + model.spm.radius_m) ** 3
    )
    # orthokinetic contributions to collision rate constant (caused by fluid motion)
    if "Freshwater" in particle.Pcompartment.Cname:
        w_den_kg_m3 = density_w_21C_kg_m3
    else:
        w_den_kg_m3 = density_seaWater_kg_m3

    MP_vSet_m_s = (
        2
        / 9
        * (float(particle.Pdensity_kg_m3) - w_den_kg_m3)
        / mu_w_21C_kg_ms
        * g_m_s2
        * (float(particle.radius_m)) ** 2
    )
    SPM_vSet_m_s = (
        2
        / 9
        * (model.spm.Pdensity_kg_m3 - w_den_kg_m3)
        / mu_w_21C_kg_ms
        * g_m_s2
        * (model.spm.radius_m) ** 2
    )
    # settling velocity. currently according to classical Stokes law. Need to include other modes and put calculation on its own, so that it can also be accessed for other processes

    k_diffSettling = (
        math.pi
        * (float(particle.radius_m) + model.spm.radius_m) ** 2
        * abs(MP_vSet_m_s - SPM_vSet_m_s)
    )
    # differential settling contributions to collision rate constant

    k_coll = k_peri + k_ortho + k_diffSettling
    # the collision rate constant

    alpha = alpha_heter[particle.Pform]

    model.spm.calc_numConc(
        concMass_mg_L=float(particle.Pcompartment.SPM_mgL), concNum_part_L=0
    )
    SPM_concNum_part_m3 = model.spm.concNum_part_m3
    k_hetAgg = float(alpha) * k_coll * SPM_concNum_part_m3
    # the pseudo first-order heteroaggregation rate constant

    k_aggBreakup = (1 / 1000000000) * k_hetAgg

    return k_aggBreakup


def advective_transport(particle, model):
    k_adv = float(particle.Pcompartment.waterFlow_m3_s) / float(
        particle.Pcompartment.Cvolume_m3
    )

    return k_adv


def mixing(particle, model):

    # Now adapted to UTOPIA's compartments and changed rates
    # k_mix has to be multiplied by the compartment volume ratio calculated with the interacting compartment volume

    # k_mix_up = (
    #     10**-2
    # )  # (1): <Handbook of Chemical Mass Transport in the Environment> Edited by Louis J. Thibodeaux, Donald Mackay (DOI: 10.1201/b10262)

    # k_mix_down = (
    #     10**-3
    # )  # (2): <Handbook on Mixing in Rivers> Edited by J.C. Rutherford (Water and Soil Miscellaneous Publication No. 26. 1981. 60pp.ISSN 0110-4705)

    # Assuming that vertical mixing for the surface of the ocean and the coast is in the 1 hour time scale. The water in the surface will take 60 min to travel 50 m, half way trhough the mix layer (100m deep). 50m/60min = 0.83 m/min= 0.0138 m/s

    # FROM OECD tool: ocean water mixing to the depths of hell (Implies residence time in the mixed layer of 100 years, Wania and Mackay GloboPOP Value, Sci Tot Env. 1995) : 1.14 * 10 ^ -4 m/h=3.167E-8 m/s

    # Assuming that vertical mixing of freshwater compartments is in the minutes timescale. The water in the surface will take 2 min to travel 5 m, half way trhough the mix layer (10m deep). 2m/5min = 0.4 m/min= 0.0067 m/s

    flowRate_mixUP_ocean_m3_s = 0.0138 * float(
        model.dict_comp["Ocean_Mixed_Water"].CsurfaceArea_m2
    )

    flowRate_mixDown_ocean_m3_s = 3.167e-8 * float(
        model.dict_comp["Ocean_Mixed_Water"].CsurfaceArea_m2
    )

    flowRate_mix_coast_m3_s = 0.0138 * float(
        model.dict_comp["Coast_Column_Water"].CsurfaceArea_m2
    )

    flowRateMix_freshWater_m3_s = 0.0067 * float(
        model.dict_comp["Bulk_Freshwater"].CsurfaceArea_m2
    )

    if particle.Pcompartment.Cname == "Ocean_Mixed_Water":
        k_mix_up = flowRate_mixUP_ocean_m3_s / float(particle.Pcompartment.Cvolume_m3)
        k_mix_down = flowRate_mixDown_ocean_m3_s / float(
            particle.Pcompartment.Cvolume_m3
        )

        k_mix = [k_mix_up, k_mix_down]
        # {"mix_up": k_mix_up, "mix_down": k_mix_down}

    elif particle.Pcompartment.Cname == "Ocean_Column_Water":
        k_mix = flowRate_mixDown_ocean_m3_s / float(particle.Pcompartment.Cvolume_m3)

    elif particle.Pcompartment.Cname == "Ocean_Surface_Water":
        k_mix = flowRate_mixUP_ocean_m3_s / float(particle.Pcompartment.Cvolume_m3)

    elif particle.Pcompartment.Cname in ["Coast_Column_Water", "Coast_Surface_Water"]:
        k_mix = flowRate_mix_coast_m3_s / float(particle.Pcompartment.Cvolume_m3)
    elif particle.Pcompartment.Cname in ["Surface_Freshwater", "Bulk_Freshwater"]:
        k_mix = flowRateMix_freshWater_m3_s / float(particle.Pcompartment.Cvolume_m3)

    else:
        print("No mixing implemented for this compartment")
        k_mix = 0

    return k_mix


def biofouling(particle, model):
    # Process by wich a biofilm is formed on the surface of the particle. The rate constant is calculated based on the growth rate of the biofilm.
    # Assumptions:
    # 1) Compartments:
    #   -Biofouling only happens in the water compartments.
    #   -Biofouling occurs at slower rates in deeper waters due to reduced light limiting the growth of the biofilm organisms (Kooi et al., 2017):Biofilm growth takes 10 days in surface water compartments and 30 days in the ocean mixed waters and coast column water and 300 days in the deep ocean

    # Values of time for biofim growth are based on experimental findings that indicate that biofilm formation takes place within days or weeks (Rummel et al., 2017)

    # 2) Aggregation state: Biofouling is modelled to occur in free and heteroaggregated particles

    t_biof_growth_days_comp = {
        "Ocean_Surface_Water": 10,
        "Ocean_Mixed_Water": 30,
        "Ocean_Column_Water": 300,
        "Coast_Surface_Water": 10,
        "Coast_Column_Water": 30,
        "Surface_Freshwater": 10,
        "Bulk_Freshwater": 30,
        "Sediment_Freshwater": 0,
        "Sediment_Ocean": 0,
        "Sediment_Coast": 0,
        "Beaches_Soil_Surface": 0,
        "Beaches_Deep_Soil": 0,
        "Background_Soil_Surface": 0,
        "Background_Soil": 0,
        "Impacted_Soil_Surface": 0,
        "Impacted_Soil": 0,
        "Air": 0,
    }
    MP_form_factor = {
        "freeMP": 1,
        "heterMP": 1,
        "biofMP": 0,
        "heterBiofMP": 0,
    }  # indicates in wich aggregation states the biofouling process is considered

    t_biof_growth_d = (
        t_biof_growth_days_comp[particle.Pcompartment.Cname]
        * MP_form_factor[particle.Pform]
    )
    if t_biof_growth_d == 0:
        k_biof = 0
    else:
        k_biof = 1 / float(t_biof_growth_d) / 24 / 60 / 60  # converted to seconds

    return k_biof


def defouling(particle, model):
    # Defouling (and its time rate measure tbiof_degrade_d) is the disintegration of the biofilm layer.

    "it can occur due to light limitation, grazing, or dissolution of carbonates in acid waters (Kooi et al., 2017).So far assumed as null due to lack of data regarding biofilm degradation times."

    # Defouling would be only modelled for the biofouled particles (biofMP and heterBiofMP?) To be decided if its depth dependent also (therefore compartment dependent)

    tbiof_degrade_d = 0
    if tbiof_degrade_d == 0:
        k_defoul = 0
    else:
        k_defoul = 1 / float(tbiof_degrade_d) / 24 / 60 / 60

    return k_defoul


def sediment_resuspension(particle, model):
    # When no depth parameter available assign transfer sediment to water rate taken from SimpleBox for Plastics model
    # Currently placeholder values. To be revisited
    resusp_dict = {
        "Sediment_Freshwater": 1e-9,
        "Sediment_Coast": 1e-10,
        "Sediment_Ocean": 1e-11,
    }

    k_resusp = resusp_dict[particle.Pcompartment.Cname]

    return k_resusp


def burial(particle, model):
    # Currenlty place holder values. To be revisited

    # When no depth parameter available assign burail rate taken from SimpleBox for Plastics model
    burial_dict = {
        "Sediment_Freshwater": 2.7e-9,
        "Sediment_Coast": 1e-9,
        "Sediment_Ocean": 5e-10,
    }

    k_burial = burial_dict[particle.Pcompartment.Cname]

    return k_burial


def soil_air_resuspension(particle, model):
    # REF: global average soil-air 6x10^-10 m/h  and max value 10^-7 m/h. Qureshi et al. (2009) ## We should include a density factor...

    sar_rate = 10e-10 / 60 / 60  # converted to m/s

    sar_rate_dict = {
        "a": sar_rate,
        "b": sar_rate,
        "c": sar_rate,
        "d": sar_rate,
        "e": sar_rate,
    }

    k_sa_reusp = sar_rate / float(particle.Pcompartment.Cdepth_m)

    return k_sa_reusp


def soil_convection(particle, model):
    # Mixing of soil particles via bioturbation and freeze/thaw cycles

    MTCsconv = 4.54e-7
    # From the OECD Tool: MTCsconv = 4.54 * 10 ^-7 (m/h)'soil side solid phase convection MTC

    k_soil_convection = (MTCsconv / (60 * 60)) / float(particle.Pcompartment.Cdepth_m)

    # if particle.Pcompartment.Cname in [
    #     "Urban_Soil_Surface",
    #     "Background_Soil_Surface",
    #     "Agricultural_Soil_Surface",
    # ]:
    #     k_soil_convection = (
    #         (C_massTransfer_m_h /(60 * 60 ))/ float(particle.Pcompartment.Cdepth_m)
    #     )
    # elif particle.Pcompartment.Cname in [
    #     "Beaches_Deep_Soil",
    #     "Impacted_Soil",
    #     "Background_Soil",
    # ]:
    #     k_soil_conv = (C_massTransfer_m_h /( 60 * 60)) / float(particle.Pcompartment.Cdepth_m)
    #     k_soil_conv_down = (
    #         (1 / 20) * (C_massTransfer_m_h /( 60 * 60 )/ float(particle.Pcompartment.Cdepth_m))
    #     )
    #     k_soil_convection = [k_soil_conv, k_soil_conv_down]

    # else:
    #     k_soil_convection = 0

    return k_soil_convection


def percolation(particle, model):
    # downwards movement of particles in soil via infiltrated water
    # # to be defined/formulated

    # k_percol = particle.Pcompartment.infiltration_capacity*particle.Pcompartment.precipitation_rate*(float(particle.Pcompartment.Cvolume_m3)/float(particle.Pcompartment.Cdepth_m))/float(particle.Pcompartment.soilPore_waterVolume_m3)

    k_percol = 0

    return k_percol


def runoff_transport(particle, model):
    # transport from top soil layers to surface waters ["Coast_Surface_Water","Surface_Freshwater"] via runoff water
    # to be formulated

    # REF: BETR global approach for MTCsoilrunoff = 2.3 * 10 ^ -8  (m/h) 'soil solids runoff rate  (Scheringer, P230)

    runooff_dict = {
        "Beaches_Soil_Surface": 2.3e-8,
        "Background_Soil_Surface": 2.3e-8,
        "Impacted_Soil_Surface": 2.3e-8,
    }
    runoff_rate = (
        runooff_dict[particle.Pcompartment.Cname]
        / float(particle.Pcompartment.Cdepth_m)
    ) / (60 * 60)

    # The total amount of runoff will be distributed into the recieving compartments according to the following matrix
    fro = np.array([[0, 1], [0, 1], [1, 0]])
    # number row corresponds to the soil emiting compartment
    soilSurf_dic = {
        "Impacted_Soil_Surface": 0,
        "Background_Soil_Surface": 1,
        "Beaches_Soil_Surface": 2,
    }
    # column number corresponds to the recieving compartment

    # In this example of fro all runoff goes to surface freshwater. To be discussed later

    k_runoff = runoff_rate * fro[soilSurf_dic[particle.Pcompartment.Cname]]
    k_runoff = k_runoff.tolist()

    return k_runoff


def beaching(particle, model):
    # Transport from surface coastal water to background soil surface.
    # We assume that beaching rate is 1/30 of the transport rate of plastic to open ocean based on https://doi.org/10.1038/s41561-023-01216-0 (Kaandorp. et al. 2023, Global mass of buoyant marine plastics dominated by large long-lived debris)

    if particle.Pcompartment.Cname == "Coast_Surface_Water":

        k_adv = float(particle.Pcompartment.waterFlow_m3_s) / float(
            particle.Pcompartment.Cvolume_m3
        )
        k_beaching = (1 / 30) * k_adv
    else:
        k_beaching = 0

    return k_beaching


def wind_trasport(particle, model):
    # diffusive transport of particles via wind speed (we should not need this process since ther is onlt one air compartment)
    # to be formulated as funcion of compartment property: wind_speed_m_s
    k_wind_transport = 0
    return k_wind_transport


def dry_deposition(particle, model):
    # particles depossition from air to soil or water compartments

    # Discuss if to use the dry depossition fractions of distribution here or move it into the fill_interactions function as done for runoff and fragments (we would contruct a dry deposition distribution matrix with the corresponding surface area ratios)

    # Based on figure 6.4 in the Handbook of Chemical Mass Transport in the Environment (2011). Dry deposition rate is size dependent

    dd_rate = 7.91e-6

    dd_rate_dict = {
        "e": dd_rate * 1000,
        "d": dd_rate * 100,
        "c": dd_rate,
        "b": dd_rate / 100,
        "a": dd_rate / 1e4,
    }

    k_dry_depossition = [
        dd_rate_dict[particle.Pcode[0]]
        * (
            float(model.dict_comp[c].CsurfaceArea_m2)
            / float(model.dict_comp["Air"].CsurfaceArea_m2)
        )
        for c in list(model.dict_comp.keys())
        if "Surface" in c
    ]
    return k_dry_depossition


def wet_deposition(particle, model):
    # Currently turned off
    # particles depossition from air to soil or water compartments via rainfall
    # wont be formulated as function of rainfall intensity but dependent on the average rain events per year. we asume that any rain event will trigger the depossition of the particles regardless of rainfall intensity
    # IN  SimpleBox for Plastics rate constant wet depossition 1.17E-1(s-1) Has to be corrected by the number of wet event and duration...so mean rate of depossition will be used
    # wd_rate=?
    # k_dry_depossition = wd_rate*float(particle.Pcompartment.CsurfaceArea_m2)/float(dict_comp["Air"].CsurfaceArea_m2)

    k_wet_depossition = 0
    return k_wet_depossition


def sea_spray_aerosol(particle, model):
    # particles resuspension from ocean and coastal surface waters to air
    # REF: REF: Qureshi et al. (2009) approach for ocean-air resuspension (10^-5 m/h) Maximum value = 10^-5 m/h!! Global average value 10^-8. We should include a density factor...
    ssa_rate = 10e-5 / 60 / 60

    k_sea_spray_aerosol = ssa_rate / float(particle.Pcompartment.Cdepth_m)

    ssa_rate_dict = {
        "a": ssa_rate,
        "b": ssa_rate,
        "c": ssa_rate,
        "d": ssa_rate,
        "e": ssa_rate,
    }

    k_sea_spray_aerosol = ssa_rate_dict[particle.Pcode[0]] / float(
        particle.Pcompartment.Cdepth_m
    )

    return k_sea_spray_aerosol


def sequestration_deep_soils(particle, model):

    # From The OECD tool: MTC3sink = 0.05 * MTCsconv (m/h)soil solids convection to the center of the earth. MTCsconv = 4.54 * 10 ^-7 (m/h)'soil side solid phase convection MTC

    MTCsconv = 4.54e-7

    # K_burial=MTC3sink (m/s) *SA (m2)/V(m3)

    k_sequestration_deep_soils = (0.05 * MTCsconv / (60 * 60)) / float(
        particle.Pcompartment.Cdepth_m
    )

    return k_sequestration_deep_soils
