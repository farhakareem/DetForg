$(document).ready(function(){
    $('.btn').on('click',function(){
        $('.spinner-border').removeClass("hide");
        //$('.btn').attr('disabled',true);
        $('.btn-txt').text("Detecting...");
    });
});