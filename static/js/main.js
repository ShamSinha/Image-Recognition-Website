$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#result1').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        $('#result1').text('');
        $('#result1').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData(document.getElementById('upload-file'));
        
        $(this).hide();
        $('.loader').show();

        const request = new XMLHttpRequest();
        request.open('POST', '/predict');
        
        request.onload = () => {

              // Extract JSON data from request
              const data = JSON.parse(request.responseText);

              // Update the result div
              document.querySelector('#result').innerHTML = data.result;
              document.querySelector('#result1').innerHTML = data.result1;
              $('#imagePreview').css('background-image', 'url(' + data.our_url + ')');
              $("#imagePreview").css("border", "0px solid #F8F8F8"); 
              $('.loader').hide();
              $('#result').fadeIn(600);
              $('#result1').fadeIn(600);
          
          }
        request.send(form_data);
        
        
    });

});
