$(document).ready(function() {
	var submitElem = $('#btnSubmit');
	if (submitElem) {
		submitElem.on('click', function() {
			$.ajax({
				url: '/etaggertest',
				data: $('#docForm').serialize(),
				method: 'POST'
			}).done(function(data) {
				if (data.success) {
					$('#info').empty();
					if(data.info) {
						$('textarea#info').text(data.info);
					}
					$('#record_table').empty();
					if(data.record) {
						var trHTML = '';
						trHTML += "<tr class=\"success\">";
						trHTML += "<th>id</th>";
						trHTML += "<th>word</th>";
						trHTML += "<th>pos</th>";
						trHTML += "<th>chk</th>";
						trHTML += "<th>tag</th>";
						trHTML += "<th>predict</th>";
						trHTML += "</tr>";
						$.each(data.record, function (i, entry) {
							$.each(entry, function (j, item) {
								trHTML += '<tr><td>' + item.id;
								trHTML += '</td><td>' + item.word;
								trHTML += '</td><td>' + item.pos;
								trHTML += '</td><td>' + item.chk;
								trHTML += '</td><td>' + item.tag;
								trHTML += '</td><td>' + item.predict;
								trHTML += '</td></tr>';
							});
							trHTML += '<tr></tr>';
						});
						$('#record_table').append(trHTML);
					}
				}
			});
		});
	}
});
