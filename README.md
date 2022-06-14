# Pull_in_calculations
This is a web-based application for generalized pull-in calculations. To launch, click this button:[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/trymhaddal/Pull_in_calculations/HEAD?urlpath=voila%2Frender%2Fmain.ipynb)

Note that it might take half a minute or so to lauch. Also, the application will time out if inactive for more than 15 minutes.

The theory used in the calculations is explained in the document "Pull-in calculation theory.pdf". Keep in mind that this application is in beta mode, so some bugs might occur. For questions, contact trym.haddal@nexans.com.

# How to use
There are four tabs; one for the geometry, one for controlling global parameters, one for options with regards to output and one for exporting the results. 

- In the Geometry tab, choose if the section is wet/dry and pipe/roller. Then select the type of geometry and relevant dimensions. Clicking "Add Section" will append this section to the sections that are already stored. To edit a section that is already added, clik "Edit Section", specify the section number, commit the changes and click "Update". You can also remove the specified section by instead clicking "Delete". An overview of the sections and the associated pull-in tension is printed below the tabs. Also plots of the geometry and minimum required pull-in tension are printed.
- In the Global Parameter tab, relevant friction factors are stored, as well as cable dry- and wet weight. These can be edited by clicking "Edit parameters" and then "Save parameters" when finished. This can be done before or after adjusting the geometry.
- In the Options tab, the user can choose the desired form of the output. The table can be chosen to be horizontal or vertical, and different options are available for how to display the units. Checkboxes control different parameters for the plots. If "Index Labels" are chosen, the user can utilize the "Text offset" slider in order to control how far the indecies are to the geometry.
- In the Export tab, the user can tick the desired outputs to generate and create a custom filename. By clicking the button "Generate files", the program will generate these outputs. Note that this does not actually download the documents, but rather generates download links under to the button after a few seconds. Download by simply clicking the links and saving.

# Known Issues
- If you input a number by text, the widget does not update until you press enter. Therefore, "Add section" or "Update section" will use the previous widget value. Therefore, make sure to always press enter after inputting number by text
