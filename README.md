<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/mvmagni/Explore_OpenCV">
    <img src="resources/HUD.jpg" alt="Logo" width="600" height="300"/>
  </a>
    <p>
    <h2 align="center">ML Image/Video Processing</h2>
    </p>

  
</div>


<!-- ABOUT THE PROJECT -->
## About The Project
<p>
This project is to setup and explore the functionality of OpenCV. Design goals are to explore functionality to create a visual interface as an assistive interface for those with ASD.  The initial goals are to target eye-hand coordination and gross motor skills. 

Utilizing mediapipe and OpenCV the system will use onscreen buttons for interaction to enable/disable features. Mediapipes built-in features such as face-mesh, hand detection, and pose detection are excellent base features to provide a visual cue for users to see the impact of their bodily movements.

<br />
Latest video on <a href="https://www.youtube.com/watch?v=RYjIu8qhYG8"> youtube </a>
</p>


<!-- ABOUT THE PROJECT -->
## Button interaction
<p> Most OpenCV interface interactions I've viewed online utilize fingers closing together in a scissor motion to "activate" the "click".  This is a more difficult fine motor skill for some so a this project will be using the z-axis depth field of mediapipe hand detection for a "tap" function similar to tapping on a touch-screen in the air</p>




<!-- ABOUT THE PROJECT -->
## Video Showcase
  * <a href="https://www.youtube.com/watch?v=RYjIu8qhYG8">Interface and feature walkthrough </a>
  * <a href="https://www.youtube.com/watch?v=jM8fBSXOj1w">20 Model performance comparison</a>

### Development tasks:
- [x] Migrate configs to external file
- [ ] Button display and pressing for interaction
- [ ] ...
</p>  
  
<p align="right">(<a href="#top">back to top</a>)</p>
</details>

## Latest screenshots
<div align="center">
  <a href="https://github.com/mvmagni/Explore_OpenCV">
    <img src="resources/config_latest.jpg" alt="Logo" width="600" height="300"/>
  </a>
    <p>
    <h2 align="center">Runtime configuration screen</h2>
    </p>
</div>

<div align="center">
  <a href="https://github.com/mvmagni/Explore_OpenCV">
    <img src="resources/runtime_latest.jpg" alt="Logo" width="600" height="300"/>
  </a>
    <p>
    <h2 align="center">System runtime</h2>
    </p>
</div>

  

<p align="right">(<a href="#top">back to top</a>)</p>


## Folder Structure
  * core - main development
  * net_configs - model configuration
  * resources - Support resources for repo (images, etc)
  
<p align="right">(<a href="#top">back to top</a>)</p>

## Resources
<a href="https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html"> OpenCV Python Tutorial</a>

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GPLv3 License. See `LICENSE.md` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Twitter - [@Ghost_in_the_NN](https://twitter.com/Ghost_in_the_NN)<br />
YouTube - [Ghostin the NN](https://www.youtube.com/channel/UC0pcRug_r2H-77KXhsImArw)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/mvmagni/Explore_OpenCV.svg?style=for-the-badge
[contributors-url]: https://github.com/mvmagni/Explore_OpenCV/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mvmagni/Explore_OpenCV.svg?style=for-the-badge
[forks-url]: https://github.com/mvmagni/Explore_OpenCV/network/members
[stars-shield]: https://img.shields.io/github/stars/mvmagni/Explore_OpenCV.svg?style=for-the-badge
[stars-url]: https://github.com/mvmagni/Explore_OpenCV/stargazers
[issues-shield]: https://img.shields.io/github/issues/mvmagni/Explore_OpenCV.svg?style=for-the-badge
[issues-url]: https://github.com/mvmagni/Explore_OpenCV/issues
[license-shield]: https://img.shields.io/github/license/mvmagni/Explore_OpenCV.svg?style=for-the-badge
[license-url]: https://github.com/mvmagni/Explore_OpenCV/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew

