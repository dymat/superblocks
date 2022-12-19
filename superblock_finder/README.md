# superblock
Finding superblock opportunities based on openstreetmap data.

This is a very minimal code description. For comprehensive simulation, a thorough code review is suggested.

Pre-processing
---------------
- Extract population data as GeoTif and store in folder (here the example is R:/Scratch/313/sven/pop_fb).
Data Source (here the example for spain):  https://data.humdata.org/search?q=High%20Resolution%20Population%20Density%20spain&ext_page_size=25&sort=score%20desc%2C%20if(gt(last_modified%2Creview_date)%2Clast_modified%2Creview_date)%20desc 

- Define the paramters in the city_metadata() function for your spatial extent.


Script execution
---------------
- Step 1: Execute the preprocess_osm.py script to obtain osm data
- Step 2: Execute the superblock script to simulat superblocks


If you are interested in using this code for academic purposes, please get in touch (sven.eggimann@empa.ch) for support and potential collaborations.


This code was used in the following academic publications:

- Eggimann, S. (2022): The potential of implementing superblocks for multifunctional street use in cities
Nature Sustainability, 5, 406–414. https://www.nature.com/articles/s41893-022-00855-2

- Eggimann S. (2022): Expanding urban green space with superblocks. Land Use Policy, 106111. https://doi.org/10.1016/j.landusepol.2022.106111