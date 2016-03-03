﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Serialization;
using System.Data;
using System.Text;
using Models.Core;
using Models.PostSimulationTools;
using APSIM.Shared.Utilities;

namespace Models
{
    /// <summary>
    /// Test interface.
    /// </summary>
    [Serializable]
    [ViewName("UserInterface.Views.GridView")]
    [PresenterName("UserInterface.Presenters.PropertyPresenter")]
    [ValidParent(ParentType = typeof(PostSimulationTools.PredictedObserved))]
    public class Tests : Model, ITestable
    {
        /// <summary>
        /// data table
        /// </summary>
        [XmlIgnore]
        public DataTable Table { get; set; }

        /// <summary>
        /// A collection of validated stats.
        /// </summary>
        [Description("An array of validated regression stats.")]
        public MathUtilities.RegrStats[] AcceptedStats { get; set; }

        /// <summary>
        /// A string containing the names of stats in the accepted values.
        /// Used for checking if the stats class has changed.
        /// </summary>
        public string AcceptedStatsName { get; set; }

        /// <summary>
        /// The name of the associated Predicted Observed node.
        /// </summary>
        public string POName { get; set; }

        /// <summary>
        /// Run tests
        /// </summary>
        /// <param name="accept">If true, the stats from this run will be written to file as the accepted stats.</param>
        /// <param name="GUIrun">If true, do not raise an exception on test failure.</param>
        public void Test(bool accept = false, bool GUIrun = false)
        {
            PredictedObserved PO = Parent as PredictedObserved;
            DataStore DS = PO.Parent as DataStore;
            MathUtilities.RegrStats[] stats;
            List<string> statNames = (new MathUtilities.RegrStats()).GetType().GetFields().Select(f => f.Name).ToList(); // use reflection, get names of stats available
            DataTable POtable = DS.GetData("*", PO.Name);
            List<string> columnNames;
            string sigIdent = "X";

            if (POtable == null)
            {
                object sim = PO.Parent;
                while (sim as Simulations == null)
                    sim = ((Model)sim).Parent;

                throw new ApsimXException(this, "Could not find PO table in " + (sim != null ? ((Simulations)sim).FileName : "<unknown>") + ". Has the simulation been run?");
            }
            columnNames = POtable.Columns.Cast<DataColumn>().Select(c => c.ColumnName).ToList(); //get list of column names
            columnNames = columnNames.Where(c => c.Contains("Observed")).ToList(); //filter names that are not pred/obs pairs
            for (int i = 0; i < columnNames.Count; i++)
                columnNames[i] = columnNames[i].Replace("Observed.", "");
            columnNames.Sort(); //ensure column names are always in the same order
            stats = new MathUtilities.RegrStats[columnNames.Count];
            List<double> x = new List<double>();
            List<double> y = new List<double>();
            string xstr, ystr;
            double xres;
            double yres;

            for (int c = 0; c < columnNames.Count; c++) //on each P/O column pair
            {
                x.Clear();
                y.Clear();
                foreach (DataRow row in POtable.Rows)
                {
                    xstr = row["Observed." + columnNames[c]].ToString();
                    ystr = row["Predicted." + columnNames[c]].ToString();
                    if (Double.TryParse(xstr, out xres) && Double.TryParse(ystr, out yres))
                    {
                        x.Add(xres);
                        y.Add(yres);
                    }
                }
                if (x.Count == 0 || y.Count == 0)
                    continue;

                stats[c] = MathUtilities.CalcRegressionStats(columnNames[c], y, x);
            }

            //remove any null stats which can occur from non-numeric columns such as dates
            List<MathUtilities.RegrStats> list = new List<MathUtilities.RegrStats>(stats);
            list.RemoveAll(l => l == null);
            stats = list.ToArray();

            //remove entries from column names
            for (int i = columnNames.Count() - 1; i >= 0; i--)
            {
                bool found = false;
                for (int j = 0; j < stats.Count(); j++)
                {
                    if (columnNames[i] == stats[j].Name)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    columnNames.RemoveAt(i);
            }

            //turn stats array into a DataTable
            //first, check if there is already an AcceptedStats array, create if not.
            //If the names don't match, then use current stats as user has dragged
            //an already existing Test to a new node.
            if (AcceptedStats == null || POName != PO.Name)
            {
                POName = PO.Name;
                AcceptedStats = stats;
                AcceptedStatsName = StringUtilities.Build(statNames, " ");
            }

            IEnumerable<string> acc = AcceptedStats[0].GetType().GetFields().Select(f => f.Name).ToList();
            //then make sure the names and order of the accepted stats are the same as the new ones.
            if (StringUtilities.Build(statNames, " ") != AcceptedStatsName)
                throw new ApsimXException(this, "Names, number or order of accepted stats do not match class MathUtilities.RegrStats. The class has probably changed.");

            Table = new DataTable("StatTests");
            Table.Columns.Add("Name", typeof(string));
            Table.Columns.Add("Variable", typeof(string));
            Table.Columns.Add("Test", typeof(string));
            Table.Columns.Add("Accepted", typeof(double));
            Table.Columns.Add("Current", typeof(double));
            Table.Columns.Add("Difference", typeof(double));
            Table.Columns.Add("Fail?", typeof(string));

            double accepted;
            double current;
            double difference;
            for (int i = 0; i < columnNames.Count; i++)
                for (int j = 1; j < statNames.Count; j++) //start at 1; we don't want Name field.
                {
                    accepted = Convert.ToDouble(AcceptedStats[i].GetType().GetField(statNames[j]).GetValue(AcceptedStats[i]));
                    current = Convert.ToDouble(stats[i].GetType().GetField(statNames[j]).GetValue(stats[i]));
                    difference = current - accepted;

                    Table.Rows.Add(PO.Name,
                        columnNames[i],
                        statNames[j],
                        accepted,
                        current,
                        difference,
                        Math.Abs(difference) > Math.Abs(accepted) * 0.01 ? "FAIL" : " ");
                }

            if (accept)
                AcceptedStats = stats;
            else
            {
                foreach (DataRow row in Table.Rows)
                    if (row["Fail?"].ToString().Equals(sigIdent))
                    {
                        if (!GUIrun)
                        {
                            object sim = PO.Parent;
                            while (sim as Simulations == null)
                                sim = ((Model)sim).Parent;

                            throw new ApsimXException(this, "Significant differences found during regression testing of " + PO.Name + " in " + (sim != null ? ((Simulations)sim).FileName : "<unknown>"));
                        }
                    }
            }
        }

        /// <summary>All simulations have run - write all tables</summary>
        /// <param name="sender">The sender.</param>
        /// <param name="e">The <see cref="EventArgs"/> instance containing the event data.</param>
        [EventSubscribe("AllCompleted")]
        private void OnAllSimulationsCompleted(object sender, EventArgs e)
        {
            Test();
        }

        /// <summary>
        /// Convert a data table to a string.
        /// http://stackoverflow.com/questions/1104121/how-to-convert-a-datatable-to-a-string-in-c
        /// Modified for multi-platform line breaks and explicit typing.
        /// </summary>
        /// <param name="dataTable"></param>
        /// <returns></returns>
        public static string ConvertDataTableToString(DataTable dataTable)
        {
            StringBuilder output = new StringBuilder();

            int[] columnsWidths = new int[dataTable.Columns.Count];
            int length;
            string text;

            // Get column widths
            foreach (DataRow row in dataTable.Rows)
            {
                for (int i = 0; i < dataTable.Columns.Count; i++)
                {
                    length = row[i].ToString().Length;
                    if (columnsWidths[i] < length)
                        columnsWidths[i] = length;
                }
            }

            // Get Column Titles
            for (int i = 0; i < dataTable.Columns.Count; i++)
            {
                length = dataTable.Columns[i].ColumnName.Length;
                if (columnsWidths[i] < length)
                    columnsWidths[i] = length;
            }

            // Write Column titles
            for (int i = 0; i < dataTable.Columns.Count; i++)
            {
                text = dataTable.Columns[i].ColumnName;
                output.Append("|" + PadCenter(text, columnsWidths[i] + 2));
            }
            output.Append("|" + Environment.NewLine + new string('=', output.Length) + Environment.NewLine);

            // Write Rows
            foreach (DataRow row in dataTable.Rows)
            {
                for (int i = 0; i < dataTable.Columns.Count; i++)
                {
                    text = row[i].ToString();
                    output.Append("|" + PadCenter(text, columnsWidths[i] + 2));
                }
                output.Append("|" + Environment.NewLine);
            }
            return output.ToString();
        }

        private static string PadCenter(string text, int maxLength)
        {
            int diff = maxLength - text.Length;
            return new string(' ', diff / 2) + text + new string(' ', (int)(diff / 2.0 + 0.5));

        }
    }
}