## 🗺️ The Treasure Map: Navigating the CSIRO Archives
Finding raw pulsar data isn't as simple as clicking "Download." The CSIRO Data Access Portal is a massive library, and without a map, you'll get lost in terabytes of data.
Here is how to navigate Project P630 (The HTRU Survey).

### The Lay of the Land (Project P630)
The High Time Resolution Universe (HTRU) survey is stored under the project code P630.

The survey is split into three distinct regions based on where the telescope points:
1. High Latitude (HighLat): Looking up/down out of the galaxy. (Clean data).
2. Medium Latitude (MedLat): Scanning the galactic strips. (Messy but rich).
3. Low Latitude (LowLat): Staring directly into the galactic center.
* Tip: We avoided LowLat. The files are often 10GB+ each. Unless you have a supercomputer, stick to High and Med latitudes.

### The Search Query
To find the list of observations, type this exact phrase into the portal:

  "Parkes observations for project P630 semester"

#### Decoding the Results
You will see hundreds of results organized by Year and Semester (e.g., 2009OCTS, 2010APRS). Usually, the first folder of a semester (the one without a specific number) contains metadata or processed "Timing" files. These are meant for this pipeline.
* You want the numbered datasets (e.g., 2009APRS_04) that contain the raw .sf (Search Mode) files.

### The Strategy
We didn't just download random files and hope for the best (that would take a lot of processing). We used a targeted approach to find files that most likely contain pulsars.
#### The Algorithm:
* Pick a Target: Find a known pulsar on the ATNF Catalogue.
* Get Coordinates: Note the Right Ascension (RA) and Declination (Dec).
* Find the Time: Check the discovery date. If it was found in 2009, look in the 2009 datasets.
* Match the Semester: Check if the observation happened in the first half (APR) or second half (OCT) of the year.

### 🎯  The "Beam Width" Rule (How to confirm a hit)
This is the most critical physics check.
The Parkes Telescope beam is narrow—roughly 0.2 degrees wide. It's like looking through a straw.
When comparing a file's coordinates to a pulsar's coordinates:
* If the difference is > 0.2°: The telescope was looking at empty space next to the pulsar. Skip it.
* If the difference is <= 0.2°: the pulsar is inside the beam.

Download that specific .sf file, feed it to the PRESTO pipeline, and let the hunt begin.
