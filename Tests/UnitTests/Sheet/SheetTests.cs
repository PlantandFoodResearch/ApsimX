﻿namespace UnitTests.Sheet
{
    using NUnit.Framework;
    using System.Collections.Generic;
    using UserInterface.Views;
    using Gtk.Sheet;
    using System.Drawing;

    [TestFixture]
    class SheetTests
    {
        /// <summary>Ensure the cell positions correctly with no scrolling.</summary>
        [Test]
        public void SheetCellPositingOkForNoScrolling()
        {
            var data = Utilities.CreateTable(new string[] {                  "A",  "B",   "C",  "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var sheet = new Sheet(new DataTableProvider(data, isReadOnly: true, units), 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 0,
                                  columnWidths: new int[] { 30, 40, 50, 60 });
            sheet.Width = 80;
            sheet.Height = 80;

            // assumes cellpadding of 2 x 10, rowHeight = 35
            // Row 0
            Assert.AreEqual(new Rectangle(0, 0, 30, 35), sheet.CalculateBounds(0, 0)); 
            Assert.AreEqual(new Rectangle(30, 0, 40, 35), sheet.CalculateBounds(1, 0));
            Assert.AreEqual(new Rectangle(70, 0, 50, 35), sheet.CalculateBounds(2, 0));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 0));

            // Row 1
            Assert.AreEqual(new Rectangle(0, 35, 30, 35), sheet.CalculateBounds(0, 1));
            Assert.AreEqual(new Rectangle(30, 35, 40, 35), sheet.CalculateBounds(1, 1));
            Assert.AreEqual(new Rectangle(70, 35, 50, 35), sheet.CalculateBounds(2, 1));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(new Rectangle(0, 70, 30, 35), sheet.CalculateBounds(0, 2));
            Assert.AreEqual(new Rectangle(30, 70, 40, 35), sheet.CalculateBounds(1, 2));
            Assert.AreEqual(new Rectangle(70, 70, 50, 35), sheet.CalculateBounds(2, 2));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 2));

            // Row 3
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }

        /// <summary>Ensure sheet view can be scrolled one cell to the right with no frozen columns.</summary>
        [Test]
        public void ScrollRightWithNoFrozenColumnsColumnsDontFit()
        {
            var data = Utilities.CreateTable(new string[] { "A", "B", "C", "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var sheet = new Sheet(new DataTableProvider(data, isReadOnly: true, units), 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 0,
                                  columnWidths: new int[] { 20, 20, 20, 20 });
            sheet.Width = 60;
            sheet.Height = 80;

            sheet.ScrollRight();

            // assumes cellpadding of 2 x 10, rowHeight = 35
            // Row 0
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 0)); // hidden
            Assert.AreEqual(new Rectangle(0, 0, 20, 35), sheet.CalculateBounds(1, 0));
            Assert.AreEqual(new Rectangle(20, 0, 20, 35), sheet.CalculateBounds(2, 0));
            Assert.AreEqual(new Rectangle(40, 0, 20, 35), sheet.CalculateBounds(3, 0));

            // Row 1
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 1)); // hidden
            Assert.AreEqual(new Rectangle(0, 35, 20, 35), sheet.CalculateBounds(1, 1));
            Assert.AreEqual(new Rectangle(20, 35, 20, 35), sheet.CalculateBounds(2, 1));
            Assert.AreEqual(new Rectangle(40, 35, 20, 35), sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 2));  // hidden
            Assert.AreEqual(new Rectangle(0, 70, 20, 35), sheet.CalculateBounds(1, 2));
            Assert.AreEqual(new Rectangle(20, 70, 20, 35), sheet.CalculateBounds(2, 2));
            Assert.AreEqual(new Rectangle(40, 70, 20, 35), sheet.CalculateBounds(3, 2));

            // Row 3
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }

        [Test]
        public void ScrollRightWithNothingToScroll()
        {
            var data = Utilities.CreateTable(new string[] { "A", "B", "C", "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var sheet = new Sheet(new DataTableProvider(data, isReadOnly: true, units), 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 0,
                                  columnWidths: new int[] { 10, 20, 20, 20 });
            sheet.Width = 80;
            sheet.Height = 80;

            sheet.ScrollRight();

            // assumes cellpadding of 2 x 10, rowHeight = 35
            // Row 0
            Assert.AreEqual(new Rectangle(0, 0, 10, 35), sheet.CalculateBounds(0, 0));
            Assert.AreEqual(new Rectangle(10, 0, 20, 35), sheet.CalculateBounds(1, 0));
            Assert.AreEqual(new Rectangle(30, 0, 20, 35), sheet.CalculateBounds(2, 0));
            Assert.AreEqual(new Rectangle(50, 0, 20, 35), sheet.CalculateBounds(3, 0));

            // Row 1
            Assert.AreEqual(new Rectangle(0, 35, 10, 35), sheet.CalculateBounds(0, 1));
            Assert.AreEqual(new Rectangle(10, 35, 20, 35), sheet.CalculateBounds(1, 1));
            Assert.AreEqual(new Rectangle(30, 35, 20, 35), sheet.CalculateBounds(2, 1));
            Assert.AreEqual(new Rectangle(50, 35, 20, 35), sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(new Rectangle(0, 70, 10, 35), sheet.CalculateBounds(0, 2));
            Assert.AreEqual(new Rectangle(10, 70, 20, 35), sheet.CalculateBounds(1, 2));
            Assert.AreEqual(new Rectangle(30, 70, 20, 35), sheet.CalculateBounds(2, 2));
            Assert.AreEqual(new Rectangle(50, 70, 20, 35), sheet.CalculateBounds(3, 2));

            // Row 3
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }

        /// <summary>
        /// Ensure sheet view can be scrolled one cell to the right with one frozen columns.
        /// </summary>
        [Test]
        public void ScrollRightWithOneFrozenColumnColumnsDontFit()
        {
            var data = Utilities.CreateTable(new string[] { "A", "B", "C", "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var sheet = new Sheet(new DataTableProvider(data, isReadOnly: true, units), 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 1,
                                  columnWidths: new int[] { 10, 20, 30, 40 });
            sheet.Width = 80;
            sheet.Height = 80;

            sheet.ScrollRight(); // This will have to scroll right 2 columns because column indexes 1,2,3 won't fit into a width of 80 pixels.

            // assumes cellpadding of 2 x 10, rowHeight = 35
            // Row 0
            Assert.AreEqual(new Rectangle(0, 0, 10, 35), sheet.CalculateBounds(0, 0));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 0));  // hidden column
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 0));  // hidden column
            Assert.AreEqual(new Rectangle(10, 0, 40, 35), sheet.CalculateBounds(3, 0));

            // Row 1
            Assert.AreEqual(new Rectangle(0, 35, 10, 35), sheet.CalculateBounds(0, 1));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 1));  // hidden column
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 1));  // hidden column
            Assert.AreEqual(new Rectangle(10, 35, 40, 35), sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(new Rectangle(0, 70, 10, 35), sheet.CalculateBounds(0, 2));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 2));  // hidden column
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 2));  // hidden column
            Assert.AreEqual(new Rectangle(10, 70, 40, 35), sheet.CalculateBounds(3, 2));

            // Row 3
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }

        /// <summary>Ensure sheet view can be scrolled one cell to the left with one frozen columns.</summary>
        [Test]
        public void ScrollLeftWithOneFrozenColumnColumnsDontFit()
        {
            var data = Utilities.CreateTable(new string[] { "A", "B", "C", "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var sheet = new Sheet(new DataTableProvider(data, isReadOnly: true, units), 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 1,
                                  columnWidths: new int[] { 10, 20, 30, 40 });
            sheet.Width = 80;
            sheet.Height = 80;

            sheet.ScrollRight(); // This will have to scroll right 2 columns because column indexes 1,2,3 won't fit into a width of 80 pixels.
            sheet.ScrollLeft();
            sheet.ScrollLeft();

            // assumes cellpadding of 2 x 10, rowHeight = 35
            Assert.AreEqual(new Rectangle(0, 0, 10, 35), sheet.CalculateBounds(0, 0));
            Assert.AreEqual(new Rectangle(10, 0, 20, 35), sheet.CalculateBounds(1, 0));
            Assert.AreEqual(new Rectangle(30, 0, 30, 35), sheet.CalculateBounds(2, 0));
            Assert.AreEqual(new Rectangle(60, 0, 40, 35), sheet.CalculateBounds(3, 0));

            // Row 1
            Assert.AreEqual(new Rectangle(0, 35, 10, 35), sheet.CalculateBounds(0, 1));
            Assert.AreEqual(new Rectangle(10, 35, 20, 35), sheet.CalculateBounds(1, 1));
            Assert.AreEqual(new Rectangle(30, 35, 30, 35), sheet.CalculateBounds(2, 1));
            Assert.AreEqual(new Rectangle(60, 35, 40, 35), sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(new Rectangle(0, 70, 10, 35), sheet.CalculateBounds(0, 2));
            Assert.AreEqual(new Rectangle(10, 70, 20, 35), sheet.CalculateBounds(1, 2));
            Assert.AreEqual(new Rectangle(30, 70, 30, 35), sheet.CalculateBounds(2, 2));
            Assert.AreEqual(new Rectangle(60, 70, 40, 35), sheet.CalculateBounds(3, 2));

            // Row 3
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }

        /// <summary>Ensure sheet view can be scrolled one row down with one heading.</summary>
        [Test]
        public void ScrollDownOneRowWithOneHeading()
        {
            var data = Utilities.CreateTable(new string[] { "A", "B", "C", "D" },
                                      new List<object[]> { new object[] {   "a1", "b1",  "c1", "d1" },
                                                           new object[] {   "a2", "b2",  "c2", "d2" },
                                                           new object[] {   "a3", "b3",  "c3", "d3" },
                                                           new object[] {   "a4", "b4",  "c4", "d4" }});
            var units = new string[] { null, "g/m2", null, null };
            var dataProvider = new DataTableProvider(data, isReadOnly: true, units);
            var sheet = new Sheet(dataProvider, 
                                  numberFrozenRows: 1,
                                  numberFrozenColumns: 1,
                                  columnWidths: new int[] { 30, 40, 50, 60 });
            sheet.Width = 80;
            sheet.Height = 80;
            sheet.RowCount = dataProvider.RowCount + 1;
            sheet.ScrollDown();

            // assumes cellpadding of 2 x 10, rowHeight = 35
            Assert.AreEqual(new Rectangle(0, 0, 30, 35), sheet.CalculateBounds(0, 0));
            Assert.AreEqual(new Rectangle(30, 0, 40, 35), sheet.CalculateBounds(1, 0));
            Assert.AreEqual(new Rectangle(70, 0, 50, 35), sheet.CalculateBounds(2, 0));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 0));

            // Row 1 - hidden
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(0, 1));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(1, 1));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(2, 1));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 1));

            // Row 2
            Assert.AreEqual(new Rectangle(0, 35, 30, 35), sheet.CalculateBounds(0, 2));
            Assert.AreEqual(new Rectangle(30, 35, 40, 35), sheet.CalculateBounds(1, 2));
            Assert.AreEqual(new Rectangle(70, 35, 50, 35), sheet.CalculateBounds(2, 2));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 2));


            // Row 3
            Assert.AreEqual(new Rectangle(0, 70, 30, 35), sheet.CalculateBounds(0, 3));
            Assert.AreEqual(new Rectangle(30, 70, 40, 35), sheet.CalculateBounds(1, 3));
            Assert.AreEqual(new Rectangle(70, 70, 50, 35), sheet.CalculateBounds(2, 3));
            Assert.AreEqual(Rectangle.Empty, sheet.CalculateBounds(3, 3));
        }
    }
}
