$(document).ready(function(){
	$("#portfolio-contant-active").mixItUp();
	$("#gallery-slider").owlCarousel({
	    paginationSpeed : 500,      
	    singleItem:true,
	    autoPlay: 4000,
	});
	$("#clients-logo").owlCarousel({
		autoPlay: 3000,
		items : 5,
		itemsDesktop : [1199,5],
		itemsDesktopSmall : [979,5],
	});

	$("#works-logo").owlCarousel({
		autoPlay: 3000,
		items : 5,
		itemsDesktop : [1199,5],
		itemsDesktopSmall : [979,5],
	});

	// Counter
	$('.counter').counterUp({
        delay: 10,
        time: 1000
    });
});

