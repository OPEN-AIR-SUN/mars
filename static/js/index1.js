window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE_1 = "./static/interpolation/stacked/transition";
var NUM_INTERP_FRAMES_1 = 75;

var interp_images_1 = [];
function preloadInterpolationImages_1() {
  for (var i = 0; i < NUM_INTERP_FRAMES_1; i++) {
    var path = INTERP_BASE_1 + '/' + String(i).padStart(6, '0') + '.png';
    interp_images_1[i] = new Image();
    interp_images_1[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image_1 = interp_images_1[i];
  image_1.ondragstart = function() { return false; };
  image_1.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper-1').empty().append(image_1);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options_1 = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels_1 = bulmaCarousel.attach('.carousel', options_1);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels_1.length; i++) {
    	// Add listener to  event
    	carousels_1[i].on('before:show', state_1 => {
    		console.log(state_1);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element_1 = document.querySelector('#my-element');
    if (element_1 && element_1.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element_1.bulmaCarousel.on('before-show', function(state_1) {
    		console.log(state_1);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES_1 - 1);

    bulmaSlider.attach();

})
