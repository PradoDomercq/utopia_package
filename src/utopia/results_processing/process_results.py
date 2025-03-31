import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd


class ResultsProcessor:
    """Provides functionalities for restructuring, analising and plotting the UTOPIA model results."""

    def __init__(self, model):
        self.model = model
        self.R = model.R
        self.Results_extended = None

    def process_results(self):
        """Reformat results from the Utopia pipeline for easier downstream analysis."""
        # Reformat results (R) dataframe
        self.R["Size_Fraction_um"] = [self.model.size_dict[x[0]] for x in self.R.index]
        self.R["MP_Form"] = [
            self.model.MP_form_dict_reverse[x[1]] for x in self.R.index
        ]
        self.R["Compartment"] = [
            self.model.comp_dict_inverse[float(x[2:-7])] for x in self.R.index
        ]

        Results = self.R[
            [
                "Compartment",
                "MP_Form",
                "Size_Fraction_um",
                "mass_g",
                "number_of_particles",
                "concentration_g_m3",
                "concentration_num_m3",
            ]
        ]
        # Calculate mass and number fractions relative to the total mass and number of particles and store in new dataframe "Results_extended"
        total_mass = sum(Results["mass_g"])
        total_number = sum(Results["number_of_particles"])
        Results_extended = Results.copy()
        Results_extended.loc[:, "mass_fraction"] = [
            x / total_mass for x in Results["mass_g"]
        ]
        Results_extended.loc[:, "number_fraction"] = [
            x / total_number for x in Results["number_of_particles"]
        ]

        mass_fraction_df = Results_extended.loc[
            :, ["Compartment", "MP_Form", "Size_Fraction_um", "mass_fraction"]
        ]

        number_fraction_df = Results_extended.loc[
            :, ["Compartment", "MP_Form", "Size_Fraction_um", "number_fraction"]
        ]
        self.Results_extended = Results_extended
        # return Results, Results_extended

    def plot_fractionDistribution_heatmaps(self, fraction):
        """Plots the mass and number fractions after they have been extracted to the Results_extended df."""
        if self.Results_extended is None:
            raise ValueError(
                "Mass and particle number fractions not extracted. Call process_results() first."
            )

        # Define the order for the MP_Form labels
        mp_form_order = [
            "freeMP",
            "heterMP",
            "biofMP",
            "heterBiofMP",
        ]  # Replace with your desired order

        # Define the order for the Compartment labels
        compartment_order = [
            "Ocean_Surface_Water",
            "Ocean_Mixed_Water",
            "Ocean_Column_Water",
            "Coast_Surface_Water",
            "Coast_Column_Water",
            "Surface_Freshwater",
            "Bulk_Freshwater",
            "Sediment_Freshwater",
            "Sediment_Ocean",
            "Sediment_Coast",
            "Beaches_Soil_Surface",
            "Beaches_Deep_Soil",
            "Background_Soil_Surface",
            "Background_Soil",
            "Impacted_Soil_Surface",
            "Impacted_Soil",
            "Air",
        ]  # Replace with your desired order

        # Pivot the DataFrame to have one row per combination of MP_Form, Compartment, and Size_Fraction_um
        pivot_table = self.Results_extended.pivot_table(
            index=["MP_Form", "Size_Fraction_um"],
            columns="Compartment",
            values=fraction,
            aggfunc="mean",
        )

        # Reorder the rows based on mp_form_order and columns based on compartment_order
        pivot_table = pivot_table.loc[mp_form_order, compartment_order]

        # Apply log scale to the pivot table
        pivot_table_log = np.log10(pivot_table)

        # Replace -inf values with NaN
        pivot_table_log.replace(-np.inf, np.nan, inplace=True)

        # Stablish a lower limit
        # Set the lower limit for the values
        lower_limit = -14
        upper_limit = np.nanmax(pivot_table_log)

        # Replace values below the lower limit with NaN
        pivot_table_log = pivot_table_log.applymap(
            lambda x: np.nan if x < lower_limit else x
        )

        # Define a custom colormap with grey color for NaN values
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad("white")

        # Plot the heatmap with logarithmic scale and custom colormap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table_log,
            cmap=cmap,
            cbar=True,
            cbar_kws={"label": "log10 (" + fraction + ") "},
            annot=False,
            linewidths=0.5,
            linecolor="grey",
            vmin=lower_limit,
            vmax=upper_limit,
        )

        # Set compartment labels to cover all size fractions underneath
        compartment_labels = pivot_table.columns
        compartment_label_positions = np.arange(len(compartment_labels)) + 0.5
        plt.xticks(
            ticks=compartment_label_positions, labels=compartment_labels, rotation=90
        )

        # Set MP_Form and Size_Fraction_um labels
        row_labels = [
            f"{mp_form} - {size_frac_um}" for mp_form, size_frac_um in pivot_table.index
        ]
        row_label_positions = np.arange(len(pivot_table.index)) + 0.5
        plt.yticks(ticks=row_label_positions, labels=row_labels, rotation=0)
        titlename = (
            "Heatmap of log10 ("
            + fraction
            + " by MP_Form, Compartment, and Size_Fraction_um"
        )
        plt.title(titlename)
        plt.xlabel("Compartment", fontsize=14)
        plt.ylabel("MP_Form - Size_Fraction_um", fontsize=14)
        plt.tight_layout()

        fig = plt.gcf()
        plt.show()

        # return fig, titlename

    def extract_results_by_compartment(self):
        if self.Results_extended is None:
            raise ValueError(
                "Mass and particle number fractions not extracted. Call process_results() first."
            )
        mass_g = []
        particle_number = []
        mass_frac_100 = []
        num_frac_100 = []
        mass_conc_g_m3 = []
        num_conc = []
        for comp in list(self.model.dict_comp.keys()):
            mass_g.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "mass_g"
                    ]
                )
            )
            particle_number.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "number_of_particles"
                    ]
                )
            )
            mass_frac_100.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "mass_fraction"
                    ]
                )
                * 100
            )
            num_frac_100.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "number_fraction"
                    ]
                )
                * 100
            )
            mass_conc_g_m3.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "concentration_g_m3"
                    ]
                )
            )
            num_conc.append(
                sum(
                    self.Results_extended[self.Results_extended["Compartment"] == comp][
                        "concentration_num_m3"
                    ]
                )
            )

        results_by_comp = pd.DataFrame(columns=["Compartments"])
        results_by_comp["Compartments"] = list(self.model.dict_comp.keys())
        results_by_comp["mass_g"] = mass_g
        results_by_comp["number_of_particles"] = particle_number
        results_by_comp["%_mass"] = mass_frac_100
        results_by_comp["%_number"] = num_frac_100
        results_by_comp["Concentration_g_m3"] = mass_conc_g_m3
        results_by_comp["Concentration_num_m3"] = num_conc

        self.results_by_comp = results_by_comp
        # return results_by_comp

    def process_all(self):
        """Runs all processing steps in order automatically."""
        self.process_results()
        for fraction in ["mass_fraction", "number_fraction"]:
            self.plot_fractionDistribution_heatmaps(fraction)
        self.extract_results_by_compartment()

    def create_rateConstants_table(self):
        df_dict = {
            "Compartment": [],
            "MP_form": [],
            "Size_Bin": [],
            "Rate_Constants": [],
        }

        for p in self.model.system_particle_object_list:
            df_dict["Compartment"].append(p.Pcompartment.Cname)
            df_dict["MP_form"].append(p.Pform)
            df_dict["Size_Bin"].append(p.Pname[:3])
            df_dict["Rate_Constants"].append(p.RateConstants)

        df = pd.DataFrame(df_dict)
        df2 = df["Rate_Constants"].apply(pd.Series)
        df = df.drop(columns="Rate_Constants")
        df3 = pd.concat([df, df2], axis=1)

        self.RC_df = df3

    def plot_rateConstants(self):
        def sum_if_list(value):
            """Returns the sum of a list if the input is a list, otherwise returns the value itself."""
            return sum(value) if isinstance(value, list) else value

        """(FIX RC for wet deposition, now its given as a list of rate constants per surface compartment only for dry deposition and wet depossition is turned off)This needs to be fixed also for the matrix of interactions and estimation of flows"""
        rateConstants_df = self.RC_df.fillna(0)
        selected_columns = rateConstants_df.columns[3:]
        data_raw = self.RC_df[selected_columns]
        selected_data = data_raw.applymap(sum_if_list)
        log_data = selected_data.applymap(lambda x: np.log10(x) if x > 0 else np.nan)

        # Violin Plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=log_data)
        # plt.yscale('log')
        plt.xticks(rotation=90)
        plt.title("Distribution of rate constants as log(k_s-1)")
        plt.show()
