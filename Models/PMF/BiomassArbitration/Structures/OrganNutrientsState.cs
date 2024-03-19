﻿using System;
using System.Collections.Generic;
using Models.Core;

namespace Models.PMF
{

    /// <summary>
    /// Daily state of flows into and out of each organ
    /// </summary>
    [Serializable]
    public class NutrientsStates : Model
    {
        /// <summary>Carbon</summary>
        public double C { get; private set; }
        /// <summary>Nitrogen</summary>
        public double N { get; private set; }
        /// <summary>Phospherous</summary>
        public double P { get; private set; }
        /// <summary>Potassium</summary>
        public double K { get; private set; }

        /// <summary>Constructor</summary>
        public NutrientsStates(double c, double n, double p, double k)
        {
            C = c;
            N = n;
            P = p;
            K = k;
        }
    }

    /// <summary>
    /// Daily state of flows into and out of each organ
    /// </summary>
    [Serializable]
    [ViewName("UserInterface.Views.PropertyView")]
    [PresenterName("UserInterface.Presenters.PropertyPresenter")]
    [ValidParent(ParentType = typeof(Organ))]
    public class OrganNutrientsState : Model, IParentOfNutrientsPoolState
    {
        /// <summary> The weight of the organ</summary>
        public NutrientPoolsState Weight
        {
            get
            {
                return Cconc > 0 ? Carbon / Cconc : new NutrientPoolsState(0, 0, 0);
            }
        }

        /// <summary> The weight of the organ</summary>
        public double Wt
        {
            get
            {
                return Weight.Total;
            }
        }

        /// <summary> The Carbon of the organ</summary>
        public double C
        {
            get
            {
                return Carbon.Total;
            }
        }

        /// <summary> The Nitrogen of the organ</summary>
        public double N
        {
            get
            {
                return Nitrogen.Total;
            }
        }

        /// <summary> The Phosphorus of the organ</summary>
        public double P
        {
            get
            {
                return Phosphorus.Total;

            }
        }

        /// <summary> The Potassium of the organ</summary>
        public double K
        {
            get
            {
                return Potassium.Total;
            }
        }

        /// <summary> The N concentration of the organ</summary>
        public double NConc
        {
            get
            {
                return Wt > 0 ? N / Wt : 0;
            }
        }

        /// <summary> The P concentration of the organ</summary>
        public double PConc
        {
            get
            {
                return Wt > 0 ? P / Wt : 0;
            }
        }

        /// <summary> The K concentration of the organ</summary>
        public double KConc
        {
            get
            {
                return Wt > 0 ? K / Wt : 0;
            }
        }


        /// <summary> The concentraion of carbon in total dry weight</summary>
        public double Cconc { get; private set; }

        /// <summary> The organs Carbon components </summary>
        public NutrientPoolsState Carbon { get; private set; }

        /// <summary> The organs Carbon components </summary>
        public NutrientPoolsState Nitrogen { get; private set; }

        /// <summary> The organs phosphorus </summary>
        public NutrientPoolsState Phosphorus { get; private set; }

        /// <summary> The organs Potasium components </summary>
        public NutrientPoolsState Potassium { get; private set; }

        /// <summary>Constructor </summary>
        public OrganNutrientsState(NutrientPoolsState carbon, NutrientPoolsState nitrogen, NutrientPoolsState phosphorus, NutrientPoolsState potassium, double cconc)
        {
            Carbon = carbon;
            Nitrogen = nitrogen;
            Phosphorus = phosphorus;
            Potassium = potassium;
            Cconc = cconc;
        }

        /// <summary>Constructor </summary>
        public OrganNutrientsState(OrganNutrientsState values, double Cconc)
        {
            Set(values, Cconc);
        }

        /// <summary>Constructor </summary>
        public void Clear()
        {
            Carbon = new NutrientPoolsState();
            Nitrogen = new NutrientPoolsState();
            Phosphorus = new NutrientPoolsState();
            Potassium = new NutrientPoolsState();
            Cconc = 0;
        }

        /// <summary>Set the current state </summary>
        public void Set(OrganNutrientsState values, double cconc)
        {
            Carbon = values.Carbon;
            Nitrogen = values.Nitrogen;
            Phosphorus = values.Phosphorus;
            Potassium = values.Potassium;
            Cconc = cconc;
        }

        /// <summary>Constructor </summary>
        public OrganNutrientsState()
        {
            Carbon = new NutrientPoolsState();
            Nitrogen = new NutrientPoolsState();
            Phosphorus = new NutrientPoolsState();
            Potassium = new NutrientPoolsState();
            Cconc = 1.0;
        }

        /// <summary>return pools divied by value</summary>
        public static OrganNutrientsState divide (OrganNutrientsState a, double b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon / b;
            ret.Nitrogen = a.Nitrogen / b;
            ret.Phosphorus = a.Phosphorus / b;
            ret.Potassium = a.Potassium / b;
            ret.Cconc = cconc;
            return ret; 

        }

        /// <summary>return pools divied by value</summary>
        public static OrganNutrientsState divide (OrganNutrientsState a, OrganNutrientsState b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon / b.Carbon;
            ret.Nitrogen = a.Nitrogen / b.Nitrogen;
            ret.Phosphorus = a.Phosphorus / b.Phosphorus;
            ret.Potassium = a.Potassium / b.Potassium;
            ret.Cconc = cconc;
            return ret;
        }

        /// <summary>return pools multiplied by value</summary>
        public static OrganNutrientsState multiply (OrganNutrientsState a, double b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon * b;
            ret.Nitrogen = a.Nitrogen * b;
            ret.Phosphorus = a.Phosphorus * b;
            ret.Potassium = a.Potassium * b;
            ret.Cconc = cconc;
            return ret;
        }

        /// <summary>return pools divied by value</summary>
        public static OrganNutrientsState multiply (OrganNutrientsState a, OrganNutrientsState b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon * b.Carbon;
            ret.Nitrogen = a.Nitrogen * b.Nitrogen;
            ret.Phosphorus = a.Phosphorus * b.Phosphorus;
            ret.Potassium = a.Potassium * b.Potassium;
            ret.Cconc = cconc;
            return ret;
        }

        /// <summary>return sum or two pools</summary>
        public static OrganNutrientsState add (OrganNutrientsState a, OrganNutrientsState b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon + b.Carbon;
            ret.Nitrogen = a.Nitrogen + b.Nitrogen;
            ret.Phosphorus = a.Phosphorus + b.Phosphorus;
            ret.Potassium = a.Potassium + b.Potassium;
            ret.Cconc = cconc;
            return ret;
        }

        /// <summary>return sum or two pools</summary>
        public static OrganNutrientsState subtract (OrganNutrientsState a, OrganNutrientsState b, double cconc)
        {
            OrganNutrientsState ret = new OrganNutrientsState();
            ret.Carbon = a.Carbon - b.Carbon;
            ret.Nitrogen = a.Nitrogen - b.Nitrogen;
            ret.Phosphorus = a.Phosphorus - b.Phosphorus;
            ret.Potassium = a.Potassium - b.Potassium;
            ret.Cconc = cconc;
            return ret;
        }
    }

    /// <summary>
    /// This is a composite biomass class, representing the sum of 1 or more biomass objects.
    /// </summary>
    [Serializable]
    [ViewName("UserInterface.Views.PropertyView")]
    [PresenterName("UserInterface.Presenters.PropertyPresenter")]
    [ValidParent(ParentType = typeof(Plant))]
    public class CompositeStates : OrganNutrientsState
    {
        private List<OrganNutrientsState> components = new List<OrganNutrientsState>();

        /// <summary>List of Organ states to include in composite state</summary>
        [Description("List of organs to agregate into composite biomass.")]
        public string[] Propertys { get; set; }

        /// <summary>Clear ourselves.</summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="EventArgs"/> instance containing the event data.</param>
        [EventSubscribe("Commencing")]
        private void OnSimulationCommencing(object sender, EventArgs e)
        {
            foreach (string PropertyName in Propertys)
            {
                OrganNutrientsState c = (OrganNutrientsState)(this.FindByPath(PropertyName)?.Value);
                if (c == null)
                    throw new Exception("Cannot find: " + PropertyName + " in composite state: " + this.Name);
            }
        }

        /// <summary>/// Add components together to give composite/// </summary>

        [EventSubscribe("PartitioningComplete")]
        public void onPartitioningComplete(object sender, EventArgs e)
        {
            Clear();
            foreach (string PropertyName in Propertys)
            {
                OrganNutrientsState c = (OrganNutrientsState)(this.FindByPath(PropertyName)?.Value);
                AddDelta(c);
            }
        }

        private void AddDelta(OrganNutrientsState delta)
        {
            double agrigatedCconc = (this.Carbon.Total + delta.Carbon.Total) / (this.Wt + delta.Wt);
            Set(OrganNutrientsState.add(this, delta,agrigatedCconc), agrigatedCconc);
        }

        /// <summary>/// The constructor </summary>
        public CompositeStates() : base() { }

        /// <summary>Writes documentation for this function by adding to the list of documentation tags.</summary>
        /// <param name="tags">The list of tags to add to.</param>
        /// <param name="headingLevel">The level (e.g. H2) of the headings.</param>
        /// <param name="indent">The level of indentation 1, 2, 3 etc.</param>
        public void Document(List<AutoDocumentation.ITag> tags, int headingLevel, int indent)
        {

            // add a heading.
            tags.Add(new AutoDocumentation.Heading(Name + " Biomass", headingLevel));

            // write description of this class.
            AutoDocumentation.DocumentModelSummary(this, tags, headingLevel, indent, false);

            // write children.
            foreach (IModel child in this.FindAllChildren<IModel>())
                AutoDocumentation.DocumentModel(child, tags, headingLevel + 1, indent);

            tags.Add(new AutoDocumentation.Paragraph(this.Name + " summarises the following biomass objects:", indent));
        }
    }
}
